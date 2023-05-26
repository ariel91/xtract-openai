from flask import Flask, render_template, request
from textblob import TextBlob

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return render_template('result.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
