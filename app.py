import tensorflow as tf
from flask import Flask,render_template,request,g
import re
import numpy as np
from tqdm.auto import tqdm

import transformers
import tensorflow
from transformers import DistilBertTokenizer, TFDistilBertModel


app=Flask(__name__)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = tf.keras.models.load_model('PSO_model_new.h5', custom_objects={"TFDistilBertModel": transformers.TFDistilBertModel})

def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256, 
        truncation=True, 
        padding='max_length', 
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }

def make_prediction(model, processed_data, classes=['LQ_CLOSE', 'LQ_EDIT', 'HQ']):
    probs = model.predict(processed_data)[0]
    return classes[np.argmax(probs)]

def remove_special_chars(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower()
    return text

def remove_digits(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_underscore(text):
    return text.replace('_', ' ')

def filter_words(text):
    return ' '.join(word for word in text.split() if len(word) > 1)
@app.route("/")
def welcome():
    return render_template("index.html")


@app.route("/stack",methods=["POST"])
def run2():
    a = request.form["title"] + " " + request.form["body"]
    a = remove_special_chars(a)
    a = remove_digits(a)
    a = remove_underscore(a)
    a = filter_words(a)
    processed_data = prepare_data(a, tokenizer)
    result = make_prediction(model, processed_data=processed_data)
    if result == "HQ" or result=="LQ_EDIT":
        
        data="This question is valid and will be accessable in Stack Overflow"
    else:
        data="This question is not valid and will be closed soon!"
        
    
    
    return render_template("index.html",new=data)



if __name__=='main':
    app.run(debug==True)
    