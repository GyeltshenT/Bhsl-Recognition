from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the model
model = pickle.load(open("rf_classifier.pkl", "rb"))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        
        # Open the image file using PIL
        image = Image.open(image_file)

        image = image.resize((64, 64))  # Resize the image to the appropriate dimensions
        image = np.array(image.convert('L'))    # Convert the image to grayscale and convert to numpy array

        image_flattened = image.flatten()

        # Make prediction using the model
        output = model.predict([image_flattened]) 

        return render_template("output.html", value=output[0])

if __name__ == "__main__":
    app.run(debug=True)
