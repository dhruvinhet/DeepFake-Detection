import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import mediapipe as mp
from flask import Flask, request, jsonify, send_file, render_template_string
from werkzeug.utils import secure_filename
import logging
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensurring
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


model_path = r"D:\Deepfake\deepfake_model_best.h5"  
model = load_model(model_path)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
pose = mp_pose.Pose(static_image_mode=True)
hands = mp_hands.Hands(static_image_mode=True)

# Box Colour for each error
ERROR_COLORS = {
    "eyes_misaligned": (255, 0, 0),
    "mouth_asymmetry": (0, 255, 0),
    "skin_color": (255, 255, 0),
    "background_inconsistency": (255, 0, 255),
}

# Image preprocessing 
def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array.astype(np.float32), None
    except Exception as e:
        return None, str(e)

# Draw bounding boxes around detected errors with label placement
def draw_error_boxes(image_path, error_boxes):
    image = cv2.imread(image_path)
    for (x1, y1, x2, y2, label, color) in error_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], "deepfake_errors_fixed.png")
    cv2.imwrite(output_path, image)
    return output_path

# Detect facial anomalies
def detect_anomalies(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Failed to load image for analysis.", [], [], {}

    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process landmarks for face, hands, and pose
    face_results = face_mesh.process(rgb_image)
    pose_results = pose.process(rgb_image)
    hands_results = hands.process(rgb_image)

    error_boxes = []
    explanation = []
    anomaly_indices = {"face": [], "hands": [], "pose": [], "color": []}
    landmarks = []

    # Face Landmarks
    if face_results.multi_face_landmarks:
        for idx, lm in enumerate(face_results.multi_face_landmarks[0].landmark):
            landmarks.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "type": "face", "idx": idx})

    # Hands Landmarks
    if hands_results.multi_hand_landmarks:
        for hand in hands_results.multi_hand_landmarks:
            for idx, lm in enumerate(hand.landmark):
                landmarks.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "type": "hand", "idx": idx})
    
    # Pose Landmarks
    if pose_results.pose_landmarks:
        for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
            landmarks.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "type": "pose", "idx": idx})

    # Add color anomaly detection
    grid_size = 50
    anomaly_regions = []  # Store detected anomaly regions
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            block = rgb_image[y:y+grid_size, x:x+grid_size]
            if block.size > 0:
                avg_color = np.mean(block, axis=(0, 1))  # Compute average color
                anomaly_regions.append(((x, y), avg_color))  # Store region color info

    # Compute color inconsistency
    color_diffs = []
    for i in range(len(anomaly_regions) - 1):
        diff = np.mean(np.abs(anomaly_regions[i][1] - anomaly_regions[i + 1][1]))
        color_diffs.append((anomaly_regions[i][0], diff))

    # Filter significant anomalies (threshold: 50)
    significant_anomalies = [region for region in color_diffs if region[1] > 50]

    # Group anomalies into general regions (forehead, left cheek, etc.)
    grouped_regions = {
        "forehead": [],
        "left cheek": [],
        "right cheek": [],
        "chin": [],
        "background": []
    }

    for (x, y), diff in significant_anomalies:
        if y < height * 0.3:
            grouped_regions["forehead"].append((x, y))
        elif y > height * 0.7:
            grouped_regions["chin"].append((x, y))
        elif x < width * 0.4:
            grouped_regions["left cheek"].append((x, y))
        elif x > width * 0.6:
            grouped_regions["right cheek"].append((x, y))
        else:
            grouped_regions["background"].append((x, y))

    # Generate concise explanations
    for region, points in grouped_regions.items():
        if points:
            explanation.append(f"⚠️ Color inconsistency detected in the {region} region.")

    return "\n".join(explanation) if explanation else "No obvious anomalies detected.", error_boxes, landmarks, anomaly_indices


def generate_heatmap(image_path, error_boxes):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Assign probability values (fakeness intensity)
    for (x1, y1, x2, y2, _, _) in error_boxes:
        heatmap[y1:y2, x1:x2] += 1  # Increase "fake probability" in detected areas

    heatmap = np.clip(heatmap, 0, 1)
    
    # Use a color map (Blue = Real, Red = Fake)
    plt.figure(figsize=(10, 10))
    sns.heatmap(heatmap, cmap="jet", alpha=0.6)  # Overlay heatmap
    plt.axis("off")
    
    heatmap_path = os.path.join(app.config['OUTPUT_FOLDER'], "heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return heatmap_path

def predict_with_explanation(image_path):
    image, error = preprocess_image(image_path)
    if image is None:
        return None, None, f"Error processing image: {error}", None, None, [], {}

    prediction = model.predict(image)
    probability_real = prediction[0][1] if prediction.shape[1] == 2 else prediction[0][0]
    label = "Real" if probability_real > 0.5 else "Deepfake"
    confidence = float(probability_real if label == "Real" else 1 - probability_real)

    if label == "Real":
        explanation = "No obvious anomalies detected."
        error_img_path = None
        heatmap_path = None
        landmarks = []
        anomaly_indices = {"face": [], "hands": [], "pose": [], "color": []}
    else:
        explanation, error_boxes, landmarks, anomaly_indices = detect_anomalies(image_path)
        error_img_path = draw_error_boxes(image_path, error_boxes) if error_boxes else None
        heatmap_path = generate_heatmap(image_path, error_boxes) if error_boxes else None

    return label, confidence, explanation, error_img_path, heatmap_path, landmarks, anomaly_indices
def apply_adversarial_noise(image_path, epsilon=0.02):
    image = cv2.imread(image_path).astype(np.float32) / 255.0
    noise = np.random.normal(scale=epsilon, size=image.shape)
    adversarial_image = np.clip(image + noise, 0, 1) * 255.0
    adversarial_path = os.path.join(app.config['OUTPUT_FOLDER'], "protected_image.png")
    cv2.imwrite(adversarial_path, adversarial_image.astype(np.uint8))
    return adversarial_path

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve the frontend
@app.route('/')
def index():
    with open("index.html", "r") as f:
        html_content = f.read()
    return render_template_string(html_content)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400
    
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)
    
    try:
        print(f"Saving file to: {image_path}")
        label, confidence, explanation, error_img_path, heatmap_path, landmarks, anomaly_indices = predict_with_explanation(image_path)
        print(f"Prediction complete: {label}, {confidence}, Landmarks count: {len(landmarks)}")
        
        response = {
            "label": label,
            "confidence": confidence,
            "explanation": explanation,
            "errorImgPath": os.path.basename(error_img_path) if error_img_path else None,
            "heatmapPath": os.path.basename(heatmap_path) if heatmap_path else None,
            "landmarks": landmarks if landmarks else [],
            "anomalyIndices": anomaly_indices
        }
        print(f"Response sent: {response}")
        return jsonify(response)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

# Serve static files (e.g., error image)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True, host='0.0.0.0', port=5000)