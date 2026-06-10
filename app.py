import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model_LSTM=load_model('next_word_lstm.h5')
model_GRU=load_model('next_word_GRU.h5')
model_EarlyStopping=load_model('next_word_lstm_model_with_early_stopping.h5')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app

st.set_page_config(page_title="Next Word Prediction With LSTM, GRU and Early Stopping", layout="wide")
st.title("Next Word Prediction With LSTM, GRU and LSTM+Early Stopping")


tab1, tab2, tab3 = st.tabs(
    [
        "Next Word Prediction With LSTM",
        "Next Word Prediction With GRU",
        "Next Word Prediction With LSTM+Early Stopping"
    ]
)
with tab1:
    st.header("Next Word Prediction With LSTM")
    input_text_LSTM=st.text_input("Enter the sequence of Words", "To be or not to", key="input_text_LSTM")
    if st.button("Predict Next Word", key="predict_button_lstm"):
        max_sequence_len = model_LSTM.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model_LSTM, tokenizer, input_text_LSTM, max_sequence_len)
        st.write(f'Next word: {next_word}')

with tab2:
    st.header("Next Word Prediction With GRU")
    input_text_GRU=st.text_input("Enter the sequence of Words", "To be or not to", key="input_text_GRU")
    if st.button("Predict Next Word", key="predict_button_gru"):
        max_sequence_len = model_GRU.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model_GRU, tokenizer, input_text_GRU, max_sequence_len)
        st.write(f'Next word: {next_word}')

with tab3:
    st.header("Next Word Prediction With LSTM+Early Stopping")
    input_text_EarlyStopping=st.text_input("Enter the sequence of Words", "To be or not to", key="input_text_early_stopping")
    if st.button("Predict Next Word", key="predict_button_early_stopping"):
        max_sequence_len = model_EarlyStopping.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model_EarlyStopping, tokenizer, input_text_EarlyStopping, max_sequence_len)
        st.write(f'Next word: {next_word}') 
