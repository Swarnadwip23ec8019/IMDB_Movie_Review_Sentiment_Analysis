import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import TextVectorization
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import nltk

# -----------------------------
# NLTK downloads
# -----------------------------
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="IMDB Sentiment Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎬 IMDB Sentiment Classifier")
st.sidebar.write("""
Welcome!  
This app predicts whether a movie review is **Positive** or **Negative** using a trained LSTM model.  
- Enter your review in the main panel.  
- Click **Analyze Sentiment**.  
- Wait a few seconds for the model to process.
""")
st.sidebar.info("The model is trained on the IMDB movie reviews dataset.")

# -----------------------------
# Main Title
# -----------------------------
st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.write("Type your movie review below to predict the sentiment:")

# -----------------------------
# Text Preprocessing
# -----------------------------
punctuation = string.punctuation
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean and preprocess text"""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = "".join([char.lower() for char in text if char not in punctuation])
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# -----------------------------
# Load Vocabulary & Vectorization Layer
# -----------------------------
vocab_file = "vocab.txt"
with open(vocab_file, encoding="utf-8") as f:
    vocab = f.read().splitlines()

max_len = 200
vectorize_layer = TextVectorization(
    max_tokens=len(vocab),
    output_mode='int',
    output_sequence_length=max_len
)
vectorize_layer.set_vocabulary(vocab)

# -----------------------------
# Load Model (without TextVectorization)
# -----------------------------
model = load_model("Sentiment_Classifier_v1_no_vectorization.keras")

# Wrap preprocessing + model for inference
full_model = Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),
    vectorize_layer,
    model
])

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area(
    "✍️ Type your review here:",
    height=150,
    placeholder="The movie was amazing and full of surprises!"
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        with st.spinner("Please wait while the sentiment is being analyzed..."):
            input_array = np.array([user_input], dtype=object)
            prediction = full_model.predict(input_array, verbose=0)[0][0]
            sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
        
        # Display results
        st.subheader("Prediction Result")
        st.markdown(f"<h3 style='color:green'>{sentiment}</h3>" if prediction > 0.5 
                    else f"<h3 style='color:red'>{sentiment}</h3>", unsafe_allow_html=True)
        st.info(f"Confidence Score: {prediction:.4f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---
**Model:** LSTM-based Sentiment Classifier  
**Dataset:** IMDB Movie Reviews  
**Framework:** TensorFlow / Keras  
""")
