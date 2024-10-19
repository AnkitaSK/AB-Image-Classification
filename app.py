from flask import Flask, render_template, request
import joblib
import numpy as np
import tensorflow as tf
import requests

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

# load model 
# Dropbox shared link (with dl=1 to force download)
dropbox_url = "https://www.dropbox.com/scl/fi/jnfeyvfvyr33ffnqt7b62/AB-CIFAR10_model.pkl?rlkey=fsk52x0ik62kn0w9v10esq9tz&st=xjttvp92&dl=1"
# Local file path where you want to save the downloaded .pkl file
local_filename = "AB-CIFAR10_model.pkl"

# Download the model file from Dropbox
try:
    response = requests.get(dropbox_url)
    response.raise_for_status()  # Check for any errors in the request

    # Save the file locally
    with open(local_filename, "wb") as file:
        file.write(response.content)
    print(f"File downloaded successfully and saved to {local_filename}")

except Exception as e:
    print(f"Error downloading file: {e}")

# Load the model after downloading the file
# Assuming the model is a .h5 or a TensorFlow model
try:
    model = joblib.load(local_filename)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")


@app.route("/", methods=["GET"])
def hello_world():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    # Check if the file is uploaded
    if "imagefile" not in request.files:
        return "No file uploaded", 400

    imagefile = request.files["imagefile"]

    # Save uploaded image to a directory
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # Load and preprocess the image
    image = load_img(
        image_path, target_size=(32, 32)
    )  # Resize to (32, 32) for CIFAR-10
    image = img_to_array(image)
    image = image.reshape(
        (1, image.shape[0], image.shape[1], image.shape[2])
    )  # Add batch dimension
    image = preprocess_input(image)

    # Make prediction
    yhat = model.predict(image)

    # CIFAR-10 class names
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Get predicted class index and confidence
    predicted_class_index = np.argmax(yhat, axis=1)[0]
    confidence = yhat[0][predicted_class_index]

    # Map the index to class label
    predicted_label = class_names[predicted_class_index]

    # Format the classification result
    classification = "%s (%.2f%%)" % (predicted_label, confidence * 100)

    return render_template("index.html", prediction=classification)


if __name__ == "__main__":
    app.run(port=3001, debug=True)
