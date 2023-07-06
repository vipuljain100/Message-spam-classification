import streamlit as st
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
def remove_emails(text):
    no_emails=re.sub(r"\S*@\S*\s?",'',text)
    return no_emails

def remove_numbers(text):
    removed_numbers=re.sub(r'\d+','',text)
    return removed_numbers

def remove_website_links(text):
    no_website_links=re.sub(r"http\S+",'',text)
    return no_website_links
def transform_text(text):
    # 1. Lowercasing
    text = text.lower()

    # 2. Tokenization
    lst = nltk.word_tokenize(text)

    # 3. Remove special characters, stopwords, and punctuation
    l1 = []
    useless_words = stopwords.words('english') + list(string.punctuation)
    for word in lst:
        if word.isalnum() == True and word not in useless_words:
            l1.append(word)

    # 4. Stemming
    l2 = []
    for word in l1:
        ps = PorterStemmer()
        l2.append(ps.stem(word))

    return " ".join(l2).strip()

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title('Spam Classifier')
sms=st.text_input('Enter the message')
stopwords_list = stopwords.words('english')
punctuation_set = set(string.punctuation)
if st.button('Predict'):
    new_text = remove_website_links(sms)
    new_text = remove_numbers(new_text)
    new_text = remove_emails(new_text)
    new_text = transform_text(new_text)

    tfidf_text = tfidf.transform([new_text]).toarray()
    result=model.predict(tfidf_text)[0]
    if result==1:
        st.write('Spam')
    else :
        st.write("Not Spam")