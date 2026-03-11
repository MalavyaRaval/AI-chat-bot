from flask import Flask, request, jsonify, render_template
import json
from chatbot_module import predict_class, get_response

app = Flask(__name__)

# Load intents for response generation
with open('AI-chat-bot/intents.json', encoding='utf-8') as f:
    intents = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'response': 'Please type something!'})
        
    ints = predict_class(user_message)
    res = get_response(ints, intents)
    return jsonify({'response': res})

if __name__ == '__main__':
    # Ensure NLTK data is downloaded once
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('wordnet')
    
    app.run(debug=True)
