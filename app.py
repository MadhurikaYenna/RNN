# ============================================
# app.py
# Streamlit RNN Sentiment Analysis App
# Updated Version (Without NLTK Errors)
# ============================================

# ============================================
# Import Libraries
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import string
import pickle
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense,
    Dropout
)

# ============================================
# Streamlit Page Configuration
# ============================================

st.set_page_config(
    page_title="RNN Sentiment Analysis",
    layout="wide"
)

st.title("🧠 RNN Sentiment Analysis App")
st.write("Text Classification using Simple RNN")

# ============================================
# Upload Dataset
# ============================================

uploaded_file = "combined_data_small.csv"

# ============================================
# Stopwords List
# ============================================

stop_words = {

    'i', 'me', 'my', 'myself', 'we', 'our',
    'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', 'her',
    'hers', 'herself', 'it', 'its',
    'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these',
    'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have',
    'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the',
    'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too',
    'very', 'can', 'will', 'just'
}

# ============================================
# Simple Tokenizer
# ============================================

def simple_tokenizer(text):
    return text.split()

# ============================================
# Text Preprocessing Function
# ============================================

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Tokenization
    words = simple_tokenizer(text)

    # Remove stopwords
    filtered_words = [

        word for word in words
        if word not in stop_words

    ]

    return " ".join(filtered_words)

# ============================================
# Proceed if File Uploaded
# ============================================

if uploaded_file is not None:

    # ========================================
    # Load Dataset
    # ========================================

    df = pd.read_csv(
        uploaded_file,
        encoding='latin1',
        engine='python',
        on_bad_lines='skip'
    )

    st.success("Dataset Loaded Successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ========================================
    # Select Columns
    # ========================================

    text_column = st.selectbox(
        "Select Text Column",
        df.columns
    )

    label_column = st.selectbox(
        "Select Label Column",
        df.columns
    )

    # ========================================
    # Text Preprocessing
    # ========================================

    texts = df[text_column].astype(str)

    cleaned_texts = texts.apply(
        preprocess_text
    )

    st.subheader("Preprocessed Text Example")

    st.write("Original Text:")
    st.write(texts.iloc[0])

    st.write("Processed Text:")
    st.write(cleaned_texts.iloc[0])

    # ========================================
    # Task 3 — Sequence Preparation
    # ========================================

    st.header("Task 3 — Sequence Preparation")

    st.write("""
    ### Why RNN Cannot Understand Raw Text

    RNN models only understand numbers.
    Text data must be converted into numerical form.

    Steps:
    1. Tokenization
    2. Word Indexing
    3. Sequence Conversion
    4. Padding

    This helps RNN learn sequential patterns.
    """)

    # ========================================
    # Tokenization
    # ========================================

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(
        cleaned_texts
    )

    vocab_size = len(
        tokenizer.word_index
    ) + 1

    # ========================================
    # Text to Sequences
    # ========================================

    sequences = tokenizer.texts_to_sequences(
        cleaned_texts
    )

    max_length = 100

    padded_sequences = pad_sequences(
        sequences,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )

    st.write("Vocabulary Size:", vocab_size)

    st.write("Sample Sequence:")
    st.write(sequences[0])

    st.write("Padded Sequence Shape:")
    st.write(padded_sequences.shape)

    # ========================================
    # Label Encoding
    # ========================================

    encoder = LabelEncoder()

    labels = encoder.fit_transform(
        df[label_column]
    )

    num_classes = len(
        np.unique(labels)
    )

    st.write("Number of Classes:", num_classes)

    # ========================================
    # Train Test Split
    # ========================================

    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences,
        labels,
        test_size=0.2,
        random_state=42
    )

    # ========================================
    # Task 4 — Build RNN Architecture
    # ========================================

    st.header("Task 4 — Build Simple RNN Architecture")

    model = Sequential([

        Embedding(
            input_dim=vocab_size,
            output_dim=128,
            input_length=max_length
        ),

        SimpleRNN(
            128,
            activation='tanh'
        ),

        Dropout(0.5),

        Dense(
            num_classes,
            activation='softmax'
        )
    ])

    # ========================================
    # Compile Model
    # ========================================

    model.compile(

        optimizer='adam',

        loss='sparse_categorical_crossentropy',

        metrics=['accuracy']
    )

    st.write("""
    ### Architecture Explanation

    - Embedding Layer:
      Converts words into vectors.

    - SimpleRNN Layer:
      Learns sequential patterns.

    - Dense Layer:
      Predicts final sentiment class.
    """)

    # ========================================
    # Task 5 — Model Training
    # ========================================

    st.header("Task 5 — Model Training")

    epochs = st.slider(
        "Select Epochs",
        1,
        20,
        5
    )

    batch_size = st.selectbox(
        "Select Batch Size",
        [16, 32, 64],
        index=1
    )

    if st.button("Train Model"):

        with st.spinner("Training Model..."):

            history = model.fit(

                X_train,
                y_train,

                epochs=epochs,

                batch_size=batch_size,

                validation_split=0.2,

                verbose=1
            )

        st.success("Model Trained Successfully")

        # ====================================
        # Task 6 — Model Evaluation
        # ====================================

        st.header("Task 6 — Model Evaluation")

        predictions = model.predict(
            X_test
        )

        predicted_labels = np.argmax(
            predictions,
            axis=1
        )

        accuracy = accuracy_score(
            y_test,
            predicted_labels
        )

        precision = precision_score(
            y_test,
            predicted_labels,
            average='weighted'
        )

        recall = recall_score(
            y_test,
            predicted_labels,
            average='weighted'
        )

        f1 = f1_score(
            y_test,
            predicted_labels,
            average='weighted'
        )

        cm = confusion_matrix(
            y_test,
            predicted_labels
        )

        st.write("### Accuracy")
        st.write(accuracy)

        st.write("### Precision")
        st.write(precision)

        st.write("### Recall")
        st.write(recall)

        st.write("### F1 Score")
        st.write(f1)

        st.write("### Confusion Matrix")
        st.write(cm)

        st.write("### Classification Report")

        st.text(
            classification_report(
                y_test,
                predicted_labels
            )
        )

        # ====================================
        # Task 7 — Sequence Understanding
        # ====================================

        st.header("Task 7 — Sequence Understanding")

        st.write("""
        ### How RNN Remembers Previous Words

        RNN processes text word-by-word.

        It remembers previous words using
        a hidden memory called Hidden State.

        Example:
        "I am very happy today"

        The model understands context
        from earlier words.

        ### Hidden State Concept

        Hidden State stores past information.

        At every step:
        - Current word is processed
        - Previous memory is combined
        - New memory is generated

        ### Sequential Learning Behavior

        Word order matters in RNN.

        Example:
        - "I am happy"
        - "I am not happy"

        Both sentences have different meanings.
        """)

        # ====================================
        # Task 8 — Real-Time Prediction
        # ====================================

        st.header("Task 8 — Real-Time Prediction")

        user_input = st.text_area(
            "Enter Sentence"
        )

        # ====================================
        # Prediction Function
        # ====================================

        def predict_sentiment(text):

            cleaned_text = preprocess_text(
                text
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

            return sentiment, confidence

        # ====================================
        # Predict Button
        # ====================================

        if st.button("Predict Sentiment"):

            sentiment, confidence = predict_sentiment(
                user_input
            )

            st.success(
                f"Predicted Sentiment: {sentiment}"
            )

            st.info(
                f"Confidence Score: {confidence:.4f}"
            )

        # ====================================
        # Sample Predictions
        # ====================================

        st.subheader("Sample Predictions")

        sample_sentences = [

            "I feel extremely happy today",

            "I am feeling depressed and lonely",

            "My anxiety is getting worse",

            "Life feels beautiful and peaceful"
        ]

        for sentence in sample_sentences:

            sentiment, confidence = predict_sentiment(
                sentence
            )

            st.write(f"Sentence: {sentence}")

            st.write(f"Prediction: {sentiment}")

            st.write(
                f"Confidence: {confidence:.4f}"
            )

            st.write("------")

        # ====================================
        # Task 9 — Save Model
        # ====================================

        st.header("Task 9 — Save Trained Model")

        if st.button("Save Model Files"):

            # Save Model
            model.save(
                "rnn_sentiment_model.h5"
            )

            # Save Tokenizer
            with open(
                "tokenizer.pkl",
                "wb"
            ) as f:

                pickle.dump(
                    tokenizer,
                    f
                )

            # Save Label Encoder
            with open(
                "label_encoder.pkl",
                "wb"
            ) as f:

                pickle.dump(
                    encoder,
                    f
                )

            st.success("""
            Files Saved Successfully

            - rnn_sentiment_model.h5
            - tokenizer.pkl
            - label_encoder.pkl
            """)

# ============================================
# END OF APP
# ============================================
