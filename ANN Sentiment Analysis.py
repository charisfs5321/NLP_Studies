# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 19:31:01 2024

@author: LENOVO
"""

import pandas as pd
import numpy as np
import spacy
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
nlp = spacy.load("xx_ent_wiki_sm")
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
dataset_path="D:/Programming/NLP/PRDECT-ID Dataset.csv"

def preprocess_dataset(text):
    processed_texts=[]
    #token_var=[]
    for sent in text:
        text=str(sent).lower()
        text=re.sub(r'(.)\1+', r'\1\1', text)
        text=re.sub(r'[^a-zA-Z\s]',' ', text)
        text=' '.join(text.split())
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_space]
        processed_texts.append(' '.join(tokens))
    return processed_texts

def predict_sentiment(sentence,model,vectorizer):
    sentence=[sentence]
    sent_vector=vectorizer.transform(sentence)
    sent_vector_dense=sent_vector.toarray()
    sentiment_predict=model.predict(sent_vector_dense)
    sentiment_predict=(sentiment_predict>0.5).astype(int)
    sentiment_mapping={0:"Negatif",1:"Positif"}
    predicted_label=sentiment_mapping[sentiment_predict[0][0]]
    return predicted_label    
    

print("Load Dataset")
dataset=pd.read_csv(dataset_path)
X=np.array(dataset['Customer Review']).astype(str)
y=np.array(dataset['Sentiment'])
print("Preprocess Text")
le=LabelEncoder()
y_encode=le.fit_transform(y)
X_vector=preprocess_dataset(X)
print("Split Dataset")
X_train,X_test,y_train,y_test=train_test_split(X_vector,y_encode,test_size=0.2,random_state=42)
print("Transform TfID")
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()
input_length=X_train_dense.shape[1]
print("Build Model")
ann_model=Sequential([
    Dense(256,activation='relu',input_shape=(input_length,)),
    Dense(1,activation='sigmoid')
                     ])

ann_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

ann_model.summary()

history=ann_model.fit(
    X_train_dense,
    y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test_dense,y_test)
    )
test_loss,test_accuracy=ann_model.evaluate(X_test_dense,y_test)
print(f"Test Accuracy: {test_accuracy}")

y_pred=ann_model.predict(X_test_dense)
y_pred = (y_pred > 0.5).astype(int) 
# Generate classification report 
print(classification_report(y_test, y_pred))
# X_vec=tokenize(X_pre)
kalimat="Produk ini mahal sekali"
prediksi_sentiment=predict_sentiment(kalimat, ann_model, vectorizer)
print(f"Kalimat: {kalimat}")
print(f"Prediksi Sentiment: {prediksi_sentiment}")