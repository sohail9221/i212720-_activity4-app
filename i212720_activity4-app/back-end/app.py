from flask import Flask, render_template, request
import numpy as np
import sqlite3

app = Flask(__name__)

# Load model weights
weights = np.load('model_weights.npy')

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0)

# Database connection
def get_db_connection():
    conn = sqlite3.connect('predictions.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form input
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Predict
        input_features = np.array([1, sepal_length, sepal_width, petal_length, petal_width])
        logits = np.dot(input_features, weights)
        probabilities = softmax(logits)
        predicted_class = np.argmax(probabilities)

        iris_classes = ['Setosa', 'Versicolor', 'Virginica']
        predicted_class_name = iris_classes[predicted_class]

        # Save to database
        conn = get_db_connection()
        conn.execute("INSERT INTO predictions (sepal_length, sepal_width, petal_length, petal_width, prediction) VALUES (?, ?, ?, ?, ?)",
                     (sepal_length, sepal_width, petal_length, petal_width, predicted_class_name))
        conn.commit()
        conn.close()

        # Render result
        return render_template('index.html', prediction_text=f'The predicted Iris species is: {predicted_class_name}')
    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
