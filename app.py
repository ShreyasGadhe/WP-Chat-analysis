from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import demoji
from transformers import pipeline

app = Flask(__name__)
demoji.download_codes()

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

def parse_whatsapp_chat(file):
    """Parse WhatsApp chat export with robust format handling"""
    content = file.read().decode('utf-8')
    pattern = re.compile(
        r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?::\d{2})?) - (.+?): (.*)$',
        re.MULTILINE
    )
    
    messages = []
    for line in content.split('\n'):
        line = line.strip()
        # Skip system messages and media notifications
        if any(keyword in line for keyword in [
            "<Media omitted>", 
            "This message was deleted",
            "end-to-end encrypted",
            "updated the message timer",
            "turned off disappearing messages"
        ]):
            continue
            
        match = pattern.match(line)
        if match:
            date, time, user, message = match.groups()
            messages.append({
                'user': user.strip(),
                'message': message.strip(),
                'timestamp': f"{date} {time}"
            })
    
    return pd.DataFrame(messages)

def preprocess(text):
    """Basic preprocessing for social media text"""
    text = demoji.replace_with_desc(text, sep=" ")  # Convert emojis to text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if not file.filename.lower().endswith('.txt'):
        return jsonify({'error': 'Only .txt files allowed'})

    try:
        df = parse_whatsapp_chat(file)
        if df.empty:
            return jsonify({'error': 'No valid messages found. Ensure export format matches WhatsApp\'s text export'})

        # Process messages in batches for efficiency
        df['clean_message'] = df['message'].apply(preprocess)
        texts = df['clean_message'].tolist()
        
        # Get predictions in batch mode
        results = sentiment_pipeline(texts, truncation=True, max_length=128)
        
        # Convert results to numerical scores
        df['sentiment'] = [
            -result['score'] if result['label'] == 'negative' else result['score']
            for result in results
        ]
        
        # Aggregate results
        user_stats = df.groupby('user').agg(
            total_messages=('message', 'count'),
            negative_messages=('sentiment', lambda x: (x < -0.2).sum()),  # -0.2 threshold works well for this model
            avg_sentiment=('sentiment', 'mean')
        ).sort_values('negative_messages', ascending=False).reset_index()

        return jsonify({
            'users': user_stats.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)