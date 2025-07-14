from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model/fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        vectorized_input = vectorizer.transform([news])
        prediction = model.predict(vectorized_input)[0]

        result = "Real News ✅" if prediction == 1 else "Fake News ❌"
        return render_template('result.html', result=result, news=news)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
