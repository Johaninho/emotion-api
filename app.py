from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from model_loader import load_emotion_model

app = Flask(__name__)

# Charger le modèle
model = load_emotion_model("mobilenet_mer_model.weights.h5")

# Étiquettes selon les classes entraînées
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" in request.files:
        image_bytes = request.files["file"].read()
    elif request.is_json and "image" in request.json:
        image_data = request.json["image"]
        image_bytes = base64.b64decode(image_data)
    else:
        return jsonify({"error": "No image provided"}), 400

    try:
        img = preprocess_image(image_bytes)
        preds = model.predict(img)[0]
        top_idx = int(np.argmax(preds))
        return jsonify({
            "emotion": emotion_labels[top_idx],
            "confidence": float(preds[top_idx])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
