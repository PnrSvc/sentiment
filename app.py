import pandas as pd
import numpy as np
import re
import streamlit as st
from transformers import pipeline
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-turkish-cased",
    tokenizer="distilbert-base-turkish-cased",
)
#Function
def data_import():
    df_comment = pd.read_csv('df_comment.csv')
    return df_comment
    
def df_sentiment(df_comment):
  preds = sentiment(df_comment)
  pred_sentiment = preds[0]["label"]
  pred_score = preds[0]["score"]
  return pred_sentiment, pred_score
def sentiment_analysis(preds):
  for i, (label, score) in enumerate(zip(preds["labels"], preds["scores"])):
        if score < 0.5:
            preds["labels"][i] = "neutral"
            preds["scores"][i] = 1.0 - score

def clean_text(text):
    text = text.encode("ascii", errors="ignore").decode(
        "ascii"
    )
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\n\n", " ", text)
    text = re.sub(r"\t", " ", text)
    text = text.strip(" ")
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation and special characters
    text = re.sub(
        " +", " ", text
    ).strip()
    return text    
iface = gr.Interface(fn = df_sentiment, 
                     inputs = "text", 
                     outputs = ['text'],
                     title = 'Sentiment Analysis', 
                     description="Get Sentiment Negative/Positive/Neutral for the given input")
                     
iface.launch(inline = False)
