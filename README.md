
# Fake News Detection System

A machine learning + Flask-based web app to detect fake news from text input.

## Features

- Text input to check if news is real or fake
- TF-IDF vectorization
- Logistic Regression Model
- Tkinter or Web UI (via Flask)

## Requirements

- Python 3.8+
- Flask
- scikit-learn
- pandas, numpy

## Run the App

```bash
pip install flask scikit-learn pandas numpy
python app.py
Visit http://localhost:5000 in your browser.

Dataset
Use train.csv with 'text' and 'label' columns for training.

label = 1 → Real

label = 0 → Fake
