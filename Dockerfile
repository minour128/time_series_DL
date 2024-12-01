FROM tensorflow/tensorflow:latest-gpu

# Utiliser une image Python comme base
FROM python:3.9-slim

WORKDIR /app

COPY . .

# Installer les dépendances
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copier le code
COPY app.py .
COPY model_cnn_gru.h5 .

# Exposer le port
EXPOSE 5000

# Démarrer l'application Flask
CMD ["python", "app.py"]

ENV FLASK_ENV=development
