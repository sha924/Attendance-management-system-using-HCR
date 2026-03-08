import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tempfile
from difflib import SequenceMatcher
import tensorflow_datasets as tfds

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "emnist_model.h5"
EMNIST_MAPPING = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"  # 47 classes mapping for EMNIST Balanced
IMG_SIZE = (28, 28)

# -------------------------
# MODEL LOADING / TRAINING
# -------------------------
@st.cache_resource
def load_or_train_model(force_train: bool = False):
    """
    Loads a saved model if available; otherwise trains a new model on EMNIST Balanced.
    Set force_train=True to retrain even if a model file exists.
    """
    if os.path.exists(MODEL_PATH) and not force_train:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            st.success("✅ Pre-trained EMNIST model loaded.")
            return model
        except Exception as e:
            st.warning(f"Could not load model ({e}). Retraining...")

    st.info("🧠 Preparing to load/train EMNIST Balanced (this may take a while).")

    # Load dataset

    ds_train, ds_test = tfds.load("emnist/balanced", split=["train", "test"], as_supervised=True, shuffle_files=True)

    def preprocess_example(img, label):
        # Convert to grayscale 28x28, normalize
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.image.rgb_to_grayscale(img)  # some EMNIST images are 1-channel already but this is safe
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds_train = ds_train.map(preprocess_example).shuffle(20000).batch(512).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess_example).batch(512).prefetch(tf.data.AUTOTUNE)

    num_classes = len(EMNIST_MAPPING)  # 47

    # Model architecture (small but effective)
    def build_model():
        inp = layers.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        out = layers.Dense(num_classes, activation="softmax")(x)

        model = models.Model(inputs=inp, outputs=out)
        return model

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Callbacks
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    # Fit
    model.fit(ds_train, validation_data=ds_test, epochs=15, callbacks=cb, verbose=1)
    model.save(MODEL_PATH)
    st.success("✅ EMNIST model trained and saved.")
    return model

# -------------------------
# IMAGE PREPROCESSING + SEGMENTATION
# -------------------------
def preprocess_for_segmentation(image_bgr):
    """Convert to grayscale, denoise, adaptive threshold, and morphological processing."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # equalize contrast
    gray = cv2.equalizeHist(gray)
    # slight blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # adaptive threshold (works better for uneven lighting)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)
    # morphological open to remove tiny noise, then dilation to join strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh

def segment_characters_from_region(thresh_img):
    """
    Finds character bounding boxes in a binary image and returns sorted list of (x, w, h, roi).
    """
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h_img, w_img = thresh_img.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter improbable boxes based on relative size to image
        if w < 3 or h < 3:
            continue
        if w > w_img * 0.9 or h > h_img * 0.9:
            continue
        # Keep boxes that likely correspond to characters/marks
        if 5 < w < w_img and 8 < h < h_img:
            roi = thresh_img[y:y+h, x:x+w]
            boxes.append((x, y, w, h, roi))
    boxes = sorted(boxes, key=lambda b: b[0])  # sort left->right
    return boxes

def prepare_char_image(roi):
    """Resize/pad ROI to 28x28 and return normalized single-channel array."""
    # Resize maintaining aspect ratio, pad to 28x28
    h, w = roi.shape
    scale = 20.0 / max(h, w)  # fit into 20x20 box
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # pad to 28x28
    padded = np.ones((28, 28), dtype=np.uint8) * 0  # background black for inverted image (we used INV)
    # center
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    padded[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    # normalize
    arr = padded.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)  # (28,28,1)
    return arr

# -------------------------
# ATTENDANCE CLASS
# -------------------------
class HandwrittenAttendance:
    def __init__(self, model, student_names=None, threshold_confidence=0.7):
        self.model = model
        self.student_names = [s.strip() for s in (student_names or []) if s.strip()]
        self.records = {}      # counts of presence
        self.total_days = 0
        self.threshold_percent = 75
        self.threshold_confidence = threshold_confidence

    def fuzzy_match(self, raw):
        if not self.student_names:
            return raw
        best = None
        best_r = 0
        for s in self.student_names:
            r = SequenceMatcher(None, raw.lower(), s.lower()).ratio()
            if r > best_r:
                best_r = r
                best = s
        return best if best_r > 0.5 else raw

    def predict_word_from_region(self, region_bgr):
        """
        Given a region (BGR) that contains a name or token, return the recognized string.
        """
        thresh = preprocess_for_segmentation(region_bgr)
        boxes = segment_characters_from_region(thresh)
        if not boxes:
            return ""
        char_images = []
        for (x, y, w, h, roi) in boxes:
            char_img = prepare_char_image(roi)
            char_images.append(char_img)
        chars_array = np.stack(char_images, axis=0)
        preds = self.model.predict(chars_array, verbose=0)
        letters = []
        for p in preds:
            conf = float(np.max(p))
            idx = int(np.argmax(p))
            if conf >= self.threshold_confidence and idx < len(EMNIST_MAPPING):
                letters.append(EMNIST_MAPPING[idx])
            else:
                letters.append("")  # blank when uncertain
        word = "".join(letters).strip()
        word = word.replace(" ", "").replace("\x00", "")
        return self.fuzzy_match(word)

    def process_image(self, bgr_image):
        if bgr_image is None:
            return
        h, w = bgr_image.shape[:2]

        # Try to detect rows by horizontal projection on a blurred grayscale image
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        proj = np.mean(blur, axis=1)  # average brightness per row

        # Find troughs (darker lines -> rows with handwriting on white paper will be dark when inverted)
        # We'll threshold relative to median
        med = np.median(proj)
        threshold = med * 0.98  # tuneable
        rows = []
        in_row = False
        start = 0
        for i, val in enumerate(proj):
            if val < threshold and not in_row:
                in_row = True
                start = i
            elif val >= threshold and in_row:
                end = i
                if end - start > 10:  # ignore very thin regions
                    rows.append((start, end))
                in_row = False
        if in_row:
            if h - start > 10:
                rows.append((start, h))

        if not rows:
            # fallback: split into fixed rows (useful for well-structured sheets)
            n_rows = 15
            row_h = h // n_rows
            rows = [(i*row_h, (i+1)*row_h) for i in range(n_rows)]

        # For each detected row, assume name column is left-most 30% (tunable)
        name_col_w = int(w * 0.30)

        for (y1, y2) in rows:
            row_img = bgr_image[y1:y2, :]
            # crop left area for name
            name_region = row_img[:, :name_col_w]
            name_text = self.predict_word_from_region(name_region)
            if name_text:
                # Normalize common OCR forms: treat single-character 'A' as 'A' (absent) only in attendance marks
                if name_text not in self.records:
                    self.records[name_text] = 0
                self.records[name_text] += 1
        # Completed one sheet/day
        self.total_days += 1

    def get_report(self):
        data = []
        # include all known students even if 0 presence
        all_names = set(self.records.keys()) | set(self.student_names)
        for name in sorted(all_names):
            present = int(self.records.get(name, 0))
            total = int(self.total_days)
            percent = round((present / total * 100) if total else 0.0, 2)
            defaulter = "Yes" if percent < self.threshold_percent else "No"
            data.append([name, present, total, percent, defaulter])
        df = pd.DataFrame(data, columns=["Name", "Present", "Total", "Percentage", "Defaulter"])
        return df

# -------------------------
# STREAMLIT UI
# -------------------------
def main():
    st.set_page_config(page_title="Handwritten Attendance HCR", layout="wide")
    st.title("🧾 Handwritten Attendance — HCR + EMNIST")
    st.write("Upload attendance sheet images (photo or scan). Provide known student names (comma separated) to improve matching.")

    col1, col2 = st.columns([2, 1])
    with col1:
        names_input = st.text_area("Known student names (comma-separated)", value="")
        student_list = [s.strip() for s in names_input.split(",") if s.strip()]
        uploaded_files = st.file_uploader("Upload attendance sheet images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    with col2:
        st.markdown("*Model options*")
        force_train = st.checkbox("Force retrain model (slow)", value=False)
        conf_slider = st.slider("Prediction confidence threshold", 0.5, 0.95, 0.7, 0.05)
        st.markdown("*Notes*: Training inside Streamlit can be slow. Provide emnist_model.h5 to skip training.")

    # Load or train model
    model = load_or_train_model(force_train)

    attendance = HandwrittenAttendance(model, student_names=student_list, threshold_confidence=conf_slider)

    if uploaded_files:
        st.info(f"Processing {len(uploaded_files)} image(s)...")
        progress = st.progress(0)
        attendance.records = {}
        attendance.total_days = 0

        for idx, f in enumerate(uploaded_files):
            try:
                # Read via PIL and convert to BGR for OpenCV
                pil = Image.open(f).convert("RGB")
                img_np = np.array(pil)[:, :, ::-1].copy()  # RGB->BGR
                attendance.process_image(img_np)
            except Exception as e:
                st.warning(f"Failed to process {getattr(f, 'name', 'uploaded file')}: {e}")
            progress.progress((idx + 1) / len(uploaded_files))

        if attendance.total_days > 0:
            df = attendance.get_report()
            st.success("✅ Attendance processed.")
            st.dataframe(df, use_container_width=True)

            # Save & download
            os.makedirs("reports", exist_ok=True)
            csv_path = os.path.join("reports", "Final_Attendance_Report.csv")
            df.to_csv(csv_path, index=False)
            with open(csv_path, "rb") as fh:
                st.download_button("⬇ Download CSV", data=fh, file_name="Final_Attendance_Report.csv", mime="text/csv")
        else:
            st.info("No attendance rows were detected. Try increasing image quality or adjusting the name-column width assumption.")

if __name__ == "__main__":
    main()
