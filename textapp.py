import streamlit as st
import joblib

model = joblib.load('model_pipeline_model.joblib')

def predict_text_with_score(model, text):
    prediction = model.predict([text])
    prediction_proba = model.predict_proba([text])
    class_labels = model.classes_
    proba_scores = {class_labels[i]: prediction_proba[0][i] for i in range(len(class_labels))}
    return prediction[0], proba_scores


st.title('Text Classification App')
user_input = st.text_area("Enter your review to see whether its good or bad:")
if st.button('Predict'):
    prediction, scores = predict_text_with_score(model, user_input)
    st.write("This review is:", prediction)
    st.write("Probability Scores:", scores)
