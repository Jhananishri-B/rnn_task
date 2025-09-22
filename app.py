import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import json
from huggingface_hub import hf_hub_download


MODEL_PATH = "headline_model_best.h5"
ARTICLE_TOKENIZER_PATH = "article_tokenizer.pkl"
HEADLINE_TOKENIZER_PATH = "headline_tokenizer.pkl"

MAX_ARTICLE_LEN = 400   
MAX_HEADLINE_LEN = 20   

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Please check the path.")
if not os.path.exists(ARTICLE_TOKENIZER_PATH) or not os.path.exists(HEADLINE_TOKENIZER_PATH):
    raise FileNotFoundError("Tokenizer files not found. Please save them after training.")

model = load_model(MODEL_PATH)

with open(ARTICLE_TOKENIZER_PATH, "rb") as f:
    article_tokenizer = pickle.load(f)

with open(HEADLINE_TOKENIZER_PATH, "rb") as f:
    headline_tokenizer = pickle.load(f)

index_to_word = {v: k for k, v in headline_tokenizer.word_index.items()}

oov_token_index = headline_tokenizer.word_index.get("<OOV>", 1)

def clean_text(text):
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def generate_headline(article):
    article = clean_text(article)
    seq = article_tokenizer.texts_to_sequences([article])
    seq = pad_sequences(seq, maxlen=MAX_ARTICLE_LEN, padding="post")

    headline_seq = [oov_token_index]

    for _ in range(MAX_HEADLINE_LEN - 1):
        decoder_input = pad_sequences([headline_seq], maxlen=MAX_HEADLINE_LEN, padding="post")
        preds = model.predict([seq, decoder_input], verbose=0)
        next_token = np.argmax(preds[0, len(headline_seq)-1, :])
        
        if next_token == 0:  
            break
        
        headline_seq.append(next_token)

    words = [index_to_word.get(idx, "") for idx in headline_seq[1:] if idx > 0]
    headline = " ".join(words).strip()
    return headline if headline else "(No headline generated)"

with gr.Blocks() as demo:
    gr.Markdown("# 📰 CNN Article → Headline Generator")
    with gr.Row():
        with gr.Column():
            article_input = gr.Textbox(
                label="Enter News Article",
                lines=10,
                placeholder="Paste your news article here..."
            )
            generate_btn = gr.Button("Generate Headline")
        with gr.Column():
            headline_output = gr.Textbox(label="Generated Headline")
    generate_btn.click(fn=generate_headline, inputs=article_input, outputs=headline_output)


if __name__ == "__main__":
    demo.launch()
