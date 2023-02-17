import sys 
import flask
from flask import Flask, render_template, jsonify, request
import processor
import time


app = Flask(__name__, static_url_path='/static')

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/chatbot', methods=['POST'])
def chatbot():
    question = request.form['question']
    # Simulate delay of 1 second
    time.sleep(1)
    response = processor.chatbot_response(question)
    return jsonify({'response': response})



if __name__ == '__main__':
    app.run(host='127.0.0.1', port='8888', debug=True)
