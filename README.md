# MNIST Handwriting Recognition

This repository contains a Flask web app and training utilities for MNIST digit recognition.

## Project structure

- `src/mnist_app/` - Flask app and model utilities
- `templates/` - HTML templates for the web app
- `static/` - frontend assets and saved history images
- `assets/dataset/` - MNIST dataset files

## Run the application

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the web server:

```bash
python run.py
```

## Train the TensorFlow model

```bash
python src/mnist_app/train.py
```
