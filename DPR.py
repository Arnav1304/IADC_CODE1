import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the TF-IDF vectorizer and the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('iadc_code_model.pkl')
    return vectorizer, model

# Define a function to preprocess text input
def preprocess_text(text):
    # Your preprocessing steps here (e.g., lowercasing, removing special characters, etc.)
    return text

# Define a function to make predictions
@st.cache
def predict_iadc_code(comment, vectorizer, model):
    # Preprocess the input text
    processed_comment = preprocess_text(comment)
    # Vectorize the preprocessed comment
    vectorized_comment = vectorizer.transform([processed_comment])
    # Make predictions
    prediction = model.predict(vectorized_comment)
    return prediction[0]

# Create the Streamlit app
def main():
    # Set the title and description
    st.title('IADC Code Predictor')
    st.write('This app predicts the IADC code based on the provided comment.')

    # Load the model
    vectorizer, model = load_model()

    # Text input for user to enter comment
    comment = st.text_area('Enter the comment here:')
    
    # Button to make predictions
    if st.button('Predict'):
        if comment:
            # Make prediction
            prediction = predict_iadc_code(comment, vectorizer, model)
            st.write('Predicted IADC Code:', prediction)
        else:
            st.write('Please enter a comment.')

# Run the app
if __name__ == '__main__':
    main()
