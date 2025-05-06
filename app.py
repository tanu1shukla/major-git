from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import tensorflow as tf

app = Flask(__name__)

# Load model and face cascade
model = tf.keras.models.load_model("facialemotionmodel.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

def extract_features(image):
    """Preprocess image for model prediction"""
    image = cv2.resize(image, (48, 48))
    return image.reshape(1, 48, 48, 1) / 255.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictemotion', methods=['POST'])
def predict_emotion():
    # Get image data from POST request
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Process base64 image
        header, encoded = data['image'].split(",", 1) if ',' in data['image'] else ('', data['image'])
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image data")

        # Convert to grayscale and detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        results = []
        for (x, y, w, h) in faces:
            try:
                # Extract and process face region
                face_roi = gray[y:y+h, x:x+w]
                processed_face = extract_features(face_roi)
                
                # Make prediction
                pred = model.predict(processed_face)
                emotion = labels[np.argmax(pred)]
                
                results.append({
                    "emotion": emotion,
                    "coordinates": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    }
                })
            except Exception as e:
                print(f"Face processing error: {str(e)}")
                continue

        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
