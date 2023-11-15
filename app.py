import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Use column layouts to organize the app content
st.title("Email/SMS Spam Classifier")
st.sidebar.header("Input")
input_sms = st.text_area("Enter the message")
st.sidebar.button("Predict")

# Use a sidebar to display the output
st.sidebar.header("Output")
if st.sidebar.button("Show"):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.markdown(f"<b>Spam</b>")
        st.markdown(f"Message: {input_sms}")
        st.markdown(f"Probability: {result}")
        st.markdown(f"Reason: {st.subheader('Reason')}")
        st.markdown(f"Explanation: {st.subheader('Explanation')}")
        st.markdown(f"Action: {st.subheader('Action')}")
        st.markdown(f"Recommendation: {st.subheader('Recommendation')}")
        st.markdown(f"Feedback: {st.subheader('Feedback')}")
    
# Use markdown to add some CSS code to style the app elements
st.markdown("""
<style>
* {
  box-sizing: border-box;
}

body {
  font-family: Arial, sans-serif;
  margin: 20px;
}

h1 {
  color: #333;
}

h2 {
  color: #666;
}

h3 {
  color: #999;
}

p {
  color: #555;
}

.b {
  font-weight: bold;
}

.i {
  font-style: italic;
}

.u {
  text-decoration: underline;
}
</style>
""")
