# ============================================
# app.py
# Streamlit RNN Sentiment Analysis Web App
# ============================================

# ============================================
# Import Libraries
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import string
import pickle
import nltk
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
# Download NLTK Resources
# ============================================

nltk.download('punkt')
nltk.download('stopwords')

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

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

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

    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):

        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(
            str.maketrans('', '', string.punctuation)
        )

        # Tokenization
        words = word_tokenize(text)

        # Stopword removal
        filtered_words = [
            word for word in words
            if word not in stop_words
        ]

        return " ".join(filtered_words)

    # ========================================
    # Apply Preprocessing
    # ========================================

    texts = df[text_column].astype(str)

    cleaned_texts = texts.apply(preprocess_text)

    st.subheader("Preprocessed Text Example")

    st.write("Original Text:")
    st.write(texts.iloc[0])

    st.write("Processed Text:")
    st.write(cleaned_texts.iloc[0])

    # ========================================
    # Tokenization & Sequence Preparation
    # ========================================

    st.header("Task 3 — Sequence Preparation")

    st.write("""
    ### Why RNN Cannot Understand Raw Text

    RNN models only understand numerical values.
    Raw text contains words and sentences that must be converted into numbers.

    Steps:
    1. Tokenization
    2. Word Indexing
    3. Sequence Conversion
    4. Padding

    This helps the RNN process sequential information.
    """)

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(cleaned_texts)

    vocab_size = len(tokenizer.word_index) + 1

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

    st.write("Padded Shape:")
    st.write(padded_sequences.shape)

    # ========================================
    # Label Encoding
    # ========================================

    encoder = LabelEncoder()

    labels = encoder.fit_transform(
        df[label_column]
    )

    num_classes = len(np.unique(labels))

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
    # Task 4 — Build Simple RNN
    # ========================================

    st.header("Task 4 — Build Simple RNN Architecture")

    model = Sequential([

        # Embedding Layer
        Embedding(
            input_dim=vocab_size,
            output_dim=128,
            input_length=max_length
        ),

        # Simple RNN Layer
        SimpleRNN(
            128,
            activation='tanh'
        ),

        Dropout(0.5),

        # Output Layer
        Dense(
            num_classes,
            activation='softmax'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    st.text(model.summary())

    st.write("""
    ### Architecture Explanation

    - Embedding Layer:
      Converts words into dense vectors.

    - SimpleRNN Layer:
      Learns sequential relationships.

    - Dense Layer:
      Predicts sentiment class.
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

        predictions = model.predict(X_test)

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

        st.write("### Evaluation Metrics")

        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1 Score:", f1)

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

        RNN processes words one-by-one in sequence.

        It stores information from previous words
        using something called a Hidden State.

        Example:
        'I am feeling very happy today'

        The model remembers earlier words while
        reading later words.

        ### Hidden State Concept

        Hidden State acts like memory.

        At every time step:
        - Current word is processed
        - Previous hidden state is combined
        - New hidden state is generated

        ### Sequential Learning Behavior

        RNN learns patterns based on order.

        Example:
        - 'I am happy'
        - 'I am not happy'

        Word order changes meaning.
        """)

        # ====================================
        # Real-Time Prediction
        # ====================================

        st.header("Task 8 — Real-Time Prediction")

        user_input = st.text_area(
            "Enter Sentence"
        )

        if st.button("Predict Sentiment"):

            def predict_sentiment(text):

                # Lowercase
                text = text.lower()

                # Remove punctuation
                text = text.translate(
                    str.maketrans(
                        '',
                        '',
                        string.punctuation
                    )
                )

                # Tokenization
                words = word_tokenize(text)

                # Stopword removal
                filtered_words = [
                    word for word in words
                    if word not in stop_words
                ]

                cleaned_text = ' '.join(
                    filtered_words
                )

                # Convert to sequence
                sequence = tokenizer.texts_to_sequences(
                    [cleaned_text]
                )

                # Padding
                padded = pad_sequences(
                    sequence,
                    maxlen=max_length,
                    padding='post',
                    truncating='post'
                )

                # Prediction
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

            sequence = tokenizer.texts_to_sequences(
                [preprocess_text(sentence)]
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
            st.write("-----")

        # ====================================
        # Task 9 — Save Model
        # ====================================

        st.header("Task 9 — Save Trained Model")

        if st.button("Save Model Files"):

            # Save Model
            model.save("rnn_sentiment_model.h5")

            # Save Tokenizer
            with open(
                "tokenizer.pkl",
                "wb"
            ) as f:
                pickle.dump(tokenizer, f)

            # Save Encoder
            with open(
                "label_encoder.pkl",
                "wb"
            ) as f:
                pickle.dump(encoder, f)

            st.success(
                "Model, Tokenizer & Encoder Saved Successfully"
            )

            st.write("""
            Files Saved:
            - rnn_sentiment_model.h5
            - tokenizer.pkl
            - label_encoder.pkl
            """)

# ============================================
# End of App
# ============================================