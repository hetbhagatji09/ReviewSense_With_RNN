import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import load_model
from utills import prediction_sentiment,preprocess_text,decode_review
import streamlit as st

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(i,2)+3 for i in words]
    padding_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padding_review

model=load_model('artifacts\my_rnn_model.h5')




word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}


st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify that it is positive or negative')


user_input=st.text_area('Message')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)
    predictions=model.predict(preprocess_input)
    Sentiment='Positive' if predictions[0][0]>0.5 else 'Negative'
    st.write(Sentiment)
    st.write(f'Prediction score is{predictions[0][0]}')