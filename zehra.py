# -*- coding: utf-8 -*-
"""
Created on Sun Jan 7 00:00:17 2024

@author: Zehra
"""

# coding: utf-8
from flask import Flask, render_template, request, make_response
import pickle
from hazm import Normalizer, Stemmer, word_tokenize, Lemmatizer
import re

app = Flask(__name__)

# Load trained model
with open('xgb.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def preprocess(text):
    text = re.sub(r'[\da-zA-Z\!\(\)\-\[\]\{\}\;\:\'\"\\\,\<\>\.\/\?\@\#\$\%\^\&\*\_\~\؟\،\٪\×\÷\»\«]', '', text)
    normalizer = Normalizer()
    text = normalizer.normalize(text)
    words = word_tokenize(text)
    stemmer = Stemmer()
    words = [stemmer.stem(word) for word in words]
    lemmatizer = Lemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

@app.route('/')
def index():
    response = make_response(render_template('index.html', prediction_text=''))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input_text']
    processed_input = preprocess(user_input)
    vectorized_input = vectorizer.transform([processed_input])  
    prediction = model.predict(vectorized_input)[0]  

    sentiment = 'happy' if prediction == 0 else 'sad'
    
    prediction_text = 'این نظر مثبته :)' if sentiment == 'happy' else 'این نظر منفیه :('
    
    return render_template('index.html', prediction_text=prediction_text, user_input=user_input)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
