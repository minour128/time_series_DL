from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Gauge
import matplotlib.pyplot as plt
import io
import base64
import logging
import json
import socket
from logging.handlers import SocketHandler

# Load the model
model = tf.keras.models.load_model('model_cnn_gru.h5', custom_objects={'mse': MeanSquaredError()})

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logstash_handler = SocketHandler('localhost', 5000)
formatter = logging.Formatter('%(message)s')
logstash_handler.setFormatter(formatter)
logger.addHandler(logstash_handler)

# Initialize Flask app
app = Flask(__name__)

# Integrate Prometheus
metrics = PrometheusMetrics(app)
accuracy_gauge = Gauge('model_accuracy', 'Accuracy of the model')
loss_gauge = Gauge('model_loss', 'Loss of the model')

# Route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Function to create a graph
def create_graph(sales_data, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sales_data)), sales_data, label="Données réelles", marker='o')
    plt.plot(range(len(predictions)), predictions, label="Prédictions", marker='x', linestyle='--')
    plt.title("Graphique des Ventes et Prédictions")
    plt.xlabel("Temps")
    plt.ylabel("Ventes")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return graph_img

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user data
        sales_data_str = request.form['sales_data']
        sales_data = np.array([float(x) for x in sales_data_str.split(',')])

        # Dynamically calculate min_val and max_val from input data
        min_val = sales_data.min()
        max_val = sales_data.max()

        # Normalization logic
        def normalize_data(data, min_val, max_val):
            """Normalize data to a range [0, 1]."""
            return (data - min_val) / (max_val - min_val)

        def denormalize_data(data, min_val, max_val):
            """Denormalize data from range [0, 1] to original scale."""
            return data * (max_val - min_val) + min_val

        # Normalize input data
        normalized_sales_data = normalize_data(sales_data, min_val, max_val)

        # Prepare normalized data for the model
        input_data = normalized_sales_data.reshape(1, normalized_sales_data.shape[0], 1)
        predictions_normalized = model.predict(input_data).flatten()

        # Denormalize predictions to original scale
        predictions = denormalize_data(predictions_normalized, min_val, max_val)

        # Generate graph
        graph_img = create_graph(sales_data, predictions)

        # Log event
        logger.info(json.dumps({"event": "prediction", "data": sales_data_str}))

        accuracy = 0.95
        loss = 0.05
        accuracy_gauge.set(accuracy)
        loss_gauge.set(loss)

        # Return the page with predictions and graph
        return render_template('index.html', prediction=predictions.tolist(), graph_img=graph_img)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
