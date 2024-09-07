def prediction_sentiment(review):
    preprocessed_input= preprocess_text(review)
    predictions=rnn_model.predict(preprocessed_input)
    sentiment= 'Positive' if predictions[0][0]>0.5 else 'Negative'
    return sentiment,predictions[0][0]
    
    
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(i,2)+3 for i in words]
    padding_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padding_review
    
def decode_review(encoded_review):
    s=[reverse_word_index.get(keys-3,'?') for keys in encoded_review]
    return ' '.join(s)
    