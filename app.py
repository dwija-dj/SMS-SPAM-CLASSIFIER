import streamlit as st
import pickle

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string


tfidf=pickle.load(open(r'vectorizer.pkl','rb'))
model=pickle.load(open(r'model.pkl','rb'))




ps=PorterStemmer()

def transform_text(text):

    text = text.lower() 
    text = nltk.word_tokenize(text) # Tokenize
    text=[word for word in text if word.isalnum()]
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text=[ps.stem(word) for word in text]
    return" ".join(text)


st.title("SMS Spam Classifier")
input_sms=st.text_area("Enter Message")
if st.button('Predict'):
    transformed_sms=transform_text(input_sms)

    vector_input=tfidf.transform([transformed_sms])

    result=model.predict(vector_input)[0]
    if result==1:
        st.header("SPAM")
    else:
        st.header("NOT A SPAM :)")
