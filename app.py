from flask import Flask, render_template, request, url_for
import string
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def load_data():
    data = joblib.load("./resources/data.pkl")
    return data


data = load_data()
y = data["Word"]

@app.route("/")
def home():
    return render_template("./resources/index.html")


@app.route('/process', methods=['POST'])
def process():
    input_text = request.form['input_text']
    input_text = preprocess_text(input_text)
    result = preprocessing(input_text)

    return render_template('./resources/result.html', result=result, desc=input_text)


def preprocessing(input_text):
    vectorizer = joblib.load("./resources/vector.pkl")
    descriptions_vec = joblib.load("./resources/desc_vector.pkl")

    input_text = preprocess_text(input_text)
    input_text_vec = vectorizer.transform([input_text])

    cosine_similarities = cosine_similarity(input_text_vec, descriptions_vec)

    top_similar_word_indices = np.argsort(cosine_similarities[0])[-50:]
    most_similar_words = [y[i] for i in top_similar_word_indices]

    return most_similar_words


if __name__ == "__main__":
    app.run(debug=True, port=5000)