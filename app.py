import gradio as gr

from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

def get_sentiment(input_text):
  return sentiment(input_text)

iface = gr.Interface(fn = get_sentiment, 
                     inputs = "text", 
                     outputs = ['text'],
                     title = 'Sentiment Analysis', 
                     description="Get Sentiment Negative/Positive for the given input")
                     
iface.launch(inline = False)
