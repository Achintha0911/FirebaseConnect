from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from firebase_admin import storage, initialize_app

initialize_app()  # Initialize Firebase

model = tf.lite.Interpreter(model_path="model.tflite")
model.allocate_tensors()

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = preprocess_image(file)  # Preprocess your image
    output = run_model(img)  # Run inference
    return jsonify({"prediction": output})

def preprocess_image(file):
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

def run_model(img):
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]["index"], img)
    model.invoke()
    output = model.get_tensor(output_details[0]["index"])
    return output.tolist()

if __name__ == "__main__":
    app.run(debug=True)
