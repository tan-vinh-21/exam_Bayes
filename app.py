from flask import Flask, request, render_template
import pickle
import string
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and features
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('features.pkl', 'rb') as file:
    features = pickle.load(file)

def convert_vector(text, features):
    # Vectorize the input text
    input_vector = np.zeros((1, len(features)))
    word_list = [word.strip(string.punctuation).lower() for word in text.split()]
    for word in word_list:
        if word in features:
            input_vector[0][features.index(word)] += 1
    return input_vector

@app.route('/')
def index():
    return render_template('index.html', result='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input text from the form
        input_text = request.form['input_text']
        
        # Vectorize the input text
        input_vector = convert_vector(input_text, features)
        
        # Predict the class
        prediction = model.predict(input_vector)[0]
        
        # Format the prediction result
        result = 'Mien Bac (B)' if prediction == 'B' else 'Mien Nam (N)'
        
        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
