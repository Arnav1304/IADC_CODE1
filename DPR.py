import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load the model and vectorizer from the disk
@st.cache_resource
def load_model():
    if os.path.exists('best_model.joblib') and os.path.exists('vectorizer.joblib'):
        best_model = joblib.load('best_model.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        return best_model, vectorizer
    else:
        raise FileNotFoundError("Model and vectorizer files not found. Train the model first.")

# Function to train model
@st.cache_resource
def train_model(data):
    comments_column = 'comments'
    iadc_code_column = 'Sub code'

    data = data.dropna(subset=[comments_column, iadc_code_column])
    comments = data[comments_column].astype(str).values
    iadc_codes = data[iadc_code_column].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(comments, iadc_codes, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression()
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l2', 'none'],
        'max_iter': [100, 200, 300, 500]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train_tfidf, y_train)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'best_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

    return best_model, vectorizer

# Function to predict IADC code
def predict_iadc_code(comment, _vectorizer, model):
    comment_tfidf = _vectorizer.transform([comment])
    prediction = model.predict(comment_tfidf)
    return prediction[0]

# Streamlit App
def main():
    st.title("IADC Code Prediction")
    st.write("Enter a comment to predict the IADC code")

    # Check if model files exist
    model_exists = os.path.exists('best_model.joblib') and os.path.exists('vectorizer.joblib')
    if model_exists:
        # Load the existing model and vectorizer
        best_model, vectorizer = load_model()
        st.write("Model loaded.")
    else:
        st.write("Model and vectorizer files not found. Please provide a dataset to train the model.")
        uploaded_file = st.file_uploader("Choose a CSV file to train the model", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, encoding='ISO-8859-1', on_bad_lines='skip')
            best_model, vectorizer = train_model(data)
            st.write("Model training completed.")

    # Text input for user comment
    user_input = st.text_area("Comment")

    if st.button("Predict"):
        if user_input:
            prediction = predict_iadc_code(user_input, vectorizer, best_model)
            st.write(f"Predicted IADC Code: {prediction}")
        else:
            st.write("Please enter a comment")

if __name__ == '__main__':
    main()
