import streamlit as st
import numpy as np
import pandas as pd
import pickle
import string
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="RNN Sentiment Analysis",
    layout="centered"
)

st.title("🧠 RNN Sentiment Analysis")

st.write("""
This application uses:
- Text Preprocessing
- Tokenization
- Sequence Padding
- Simple RNN Architecture
- Real-Time Sentiment Prediction
""")

# ==========================================
# LOAD MODEL & FILES
# ==========================================

model = load_model(
    "rnn_sentiment_model.keras"
)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ==========================================
# STOPWORDS
# ==========================================

stop_words = {

    'i','me','my','myself','we','our',
    'ours','ourselves','you','your',
    'yours','yourself','yourselves',
    'he','him','his','himself','she',
    'her','hers','herself','it','its',
    'itself','they','them','their',
    'theirs','themselves','what',
    'which','who','whom','this',
    'that','these','those','am',
    'is','are','was','were','be',
    'been','being','have','has',
    'had','having','do','does',
    'did','doing','a','an','the',
    'and','but','if','or','because',
    'as','until','while','of','at',
    'by','for','with','about',
    'against','between','into',
    'through','during','before',
    'after','above','below','to',
    'from','up','down','in','out',
    'on','off','over','under',
    'again','further','then','once',
    'here','there','when','where',
    'why','how','all','any','both',
    'each','few','more','most',
    'other','some','such','no',
    'nor','not','only','own',
    'same','so','than','too','very'
}

# ==========================================
# PREPROCESS FUNCTION
# ==========================================

def preprocess_text(text):

    # lowercase
    text = text.lower()

    # remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # tokenize
    words = text.split()

    # remove stopwords
    filtered_words = [

        word for word in words
        if word not in stop_words

    ]

    return " ".join(filtered_words)

# ==========================================
# TASK EXPLANATIONS
# ==========================================

st.header("📘 Sequence Understanding")

st.write("""
### Why RNN Cannot Understand Raw Text

RNN models only understand numerical values.
Text must be converted into sequences.

Steps:
1. Tokenization
2. Word Indexing
3. Numerical Sequences
4. Padding

### Hidden State Concept

RNN stores previous information using hidden states.

### Sequential Learning

RNN learns word order and context.

Example:
- I am happy
- I am not happy

Both have different meanings.
""")

# ==========================================
# USER INPUT
# ==========================================

st.header("💬 Real-Time Prediction")

user_input = st.text_area(
    "Enter Sentence"
)

# ==========================================
# PREDICTION
# ==========================================

max_length = 100

if st.button("Predict Sentiment"):

    cleaned_text = preprocess_text(
        user_input
    )

    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )

    prediction = model.predict(
        padded
    )

    predicted_class = np.argmax(
        prediction
    )

    sentiment = encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = np.max(prediction)

    st.success(
        f"Predicted Sentiment: {sentiment}"
    )

    st.info(
        f"Confidence Score: {confidence:.4f}"
    )

# ==========================================
# SAMPLE TESTS
# ==========================================

st.header("🧪 Sample Predictions")

sample_sentences = [

    "I feel extremely happy today",

    "I am feeling depressed and lonely",

    "My anxiety is getting worse",

    "Life feels beautiful and peaceful"
]

for sentence in sample_sentences:

    cleaned_text = preprocess_text(
        sentence
    )

    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding='post'
    )

    prediction = model.predict(
        padded
    )

    predicted_class = np.argmax(
        prediction
    )

    sentiment = encoder.inverse_transform(
        [predicted_class]
    )[0]

    st.write(f"Sentence: {sentence}")

    st.write(f"Prediction: {sentiment}")

    st.write("---")
