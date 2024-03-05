from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('trained_model.h5')

# Define the class labels
class_labels = ['Organic', 'Recyclable']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)
    predicted_class = class_labels[int(np.round(prediction[0]))]

    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = 'uploads/' + file.filename
            file.save(file_path)
            prediction = predict_image(file_path)
            return render_template('result.html', filename=file.filename, prediction=prediction)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
