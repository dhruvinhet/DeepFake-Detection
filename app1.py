import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained deepfake detection model
model_path = r"D:\deepfake_model_best.h5"
model = load_model(model_path)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# Define colors for different errors
ERROR_COLORS = {
    "glasses": (0, 0, 255),  # Red
    "eyes_misaligned": (255, 0, 0),  # Blue
    "mouth_asymmetry": (0, 255, 0),  # Green
    "skin_color": (255, 255, 0),  # Cyan
}

# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# Draw bounding boxes around detected errors with improved label placement
def draw_error_boxes(image_path, error_boxes):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    used_positions = {}  # Dictionary to track used y-positions for each x-range

    for error in error_boxes:
        x1, y1, x2, y2, label, color = error
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Calculate text size and position
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_x = x1
        text_y = y1 - 10  # Default position above the box

        # Check for overlap and adjust y-position
        x_range = (text_x, text_x + w)
        if x_range not in used_positions:
            used_positions[x_range] = []
        
        while any(abs(text_y - pos) < h + 5 for pos in used_positions[x_range]):
            text_y -= h + 5  # Move up if overlapping with any existing label
        
        used_positions[x_range].append(text_y)

        # Ensure text_y doesn't go off the image
        text_y = max(5, min(text_y, height - h - 5))

        # Add semi-transparent background for readability
        cv2.rectangle(image, (text_x, text_y - h - 3), (text_x + w + 3, text_y + 3), color, -1)
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    output_path = "deepfake_errors_fixed.png"
    cv2.imwrite(output_path, image)
    return output_path

# Detect facial anomalies
def detect_facial_anomalies(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return "No face detected. Image might be heavily manipulated.", []

    face_landmarks = results.multi_face_landmarks[0]

    # Extract key facial points
    left_eye = (int(face_landmarks.landmark[33].x * width), int(face_landmarks.landmark[33].y * height))
    right_eye = (int(face_landmarks.landmark[263].x * width), int(face_landmarks.landmark[263].y * height))
    mouth_left = (int(face_landmarks.landmark[61].x * width), int(face_landmarks.landmark[61].y * height))
    mouth_right = (int(face_landmarks.landmark[291].x * width), int(face_landmarks.landmark[291].y * height))

    explanation = []
    error_boxes = []

    # Check if glasses are present before checking asymmetry
    left_eye_region = rgb_image[left_eye[1] - 5:left_eye[1] + 5, left_eye[0] - 5:left_eye[0] + 5] if left_eye[1] - 5 >= 0 and left_eye[0] - 5 >= 0 else np.array([])
    right_eye_region = rgb_image[right_eye[1] - 5:right_eye[1] + 5, right_eye[0] - 5:right_eye[0] + 5] if right_eye[1] - 5 >= 0 and right_eye[0] - 5 >= 0 else np.array([])

    left_glass_presence = np.mean(left_eye_region) if left_eye_region.size > 0 else 0
    right_glass_presence = np.mean(right_eye_region) if right_eye_region.size > 0 else 0

    if left_glass_presence > 50 or right_glass_presence > 50:
        if abs(left_glass_presence - right_glass_presence) > 30:
            explanation.append("🕶️ Glasses detected only on one eye, which is a common deepfake error.")
            error_boxes.append((left_eye[0], left_eye[1], right_eye[0], right_eye[1], "One-Sided Glasses", ERROR_COLORS["glasses"]))

    # Check for unnatural facial asymmetry
    if abs(left_eye[1] - right_eye[1]) > 8:
        explanation.append("⚠️ Eyes appear misaligned, indicating possible blending issues.")
        error_boxes.append((left_eye[0], left_eye[1], right_eye[0], right_eye[1], "Eye Misalignment", ERROR_COLORS["eyes_misaligned"]))

    # Check for mouth distortion
    if abs(mouth_left[1] - mouth_right[1]) > 10:
        explanation.append("⚠️ Mouth asymmetry detected, which is often a sign of AI-generated manipulation.")
        error_boxes.append((mouth_left[0], mouth_left[1], mouth_right[0], mouth_right[1], "Mouth Asymmetry", ERROR_COLORS["mouth_asymmetry"]))

    # Skin color inconsistency detection
    skin_sample1 = rgb_image[height//3:height//3+10, width//3:width//3+10] if height//3+10 < height and width//3+10 < width else np.array([])
    skin_sample2 = rgb_image[2*height//3:2*height//3+10, 2*width//3:2*width//3+10] if 2*height//3+10 < height and 2*width//3+10 < width else np.array([])

    if skin_sample1.size > 0 and skin_sample2.size > 0:
        skin_diff = np.abs(np.mean(skin_sample1) - np.mean(skin_sample2))
        if skin_diff > 30:
            explanation.append("🎭 Skin color inconsistency detected, which is a deepfake blending artifact.")
            error_boxes.append((width//3, height//3, 2*width//3, 2*height//3, "Skin Color Issue", ERROR_COLORS["skin_color"]))

    return "\n".join(explanation) if explanation else "No obvious facial anomalies detected.", error_boxes

# Predict deepfake and provide explanation
def predict_with_explanation(image_path, model):
    image = preprocess_image(image_path)
    prediction = model.predict(image)

    probability_real = prediction[0][1] if prediction.shape[1] == 2 else prediction[0][0]
    label = "Real" if probability_real > 0.5 else "Deepfake"
    confidence = probability_real if label == "Real" else 1 - probability_real

    print(f"Prediction: {label} (Confidence: {confidence:.2f})")

    if label == "Real":
        explanation = "No obvious facial anomalies detected."
    else:
        explanation, error_boxes = detect_facial_anomalies(image_path)

    print(f"Explanation:\n{explanation}")

    if label == "Deepfake":
        error_img_path = draw_error_boxes(image_path, error_boxes)
        print(f"Error visualization saved as: {error_img_path}")

    return label, confidence, explanation

# Run the detection
if __name__ == "__main__":
    image_path = "1.jpg"  
    label, confidence, explanation = predict_with_explanation(image_path, model)