from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import demoji
from transformers import pipeline
from io import StringIO
import os

app = Flask(__name__)
demoji.download_codes()

# Load multilingual sentiment analysis model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="xlm-roberta-large-xnli",
    tokenizer="xlm-roberta-large-xnli"
)

def parse_whatsapp_chat(file):
    content = file.read().decode('utf-8')
    pattern = re.compile(r'\[(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}:\d{2})\] (.+?): (.*)')
    messages = []
    
    for line in content.split('\n'):
        match = pattern.match(line)
        if match and not line.startswith('You deleted this message'):
            date, time, user, message = match.groups()
            messages.append({
                'user': user.strip(),
                'message': message.strip(),
                'timestamp': f"{date} {time}"
            })
    return pd.DataFrame(messages)

def preprocess(text):
    text = demoji.replace_with_desc(text, sep=" ")
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.lower()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file'})
    
    # Parse WhatsApp chat
    df = parse_whatsapp_chat(file)
    
    # Preprocess and analyze sentiment
    df['clean_message'] = df['message'].apply(preprocess)
    df['sentiment'] = df['clean_message'].apply(
        lambda x: sentiment_pipeline(x[:512])[0]['score'] * (-1 if sentiment_pipeline(x[:512])[0]['label'] == 'negative' else 1)
    )
    
    # Aggregate by user
    user_stats = df.groupby('user').agg(
        total_messages=('message', 'count'),
        negative_score=('sentiment', lambda x: (x < -0.5).sum()),
        avg_sentiment=('sentiment', 'mean')
    ).sort_values('negative_score', ascending=False).reset_index()
    
    # Anonymize user names
    user_stats['user'] = [f'User {i+1}' for i in range(len(user_stats))]
    
    return jsonify({
        'users': user_stats.to_dict(orient='records')
    })

if __name__ == '__main__':
    app.run(debug=True)