
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"

import streamlit as st
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def sentences_similar(sentence1, sentence2):
    emb1 = model.encode(sentence1)
    emb2 = model.encode(sentence2)
    cos_sim = util.cos_sim(emb1, emb2)
    return cos_sim

st.title("Sentence Similarity Checker")

sentence1 = st.text_input("Enter Sentence 1", "")
sentence2 = st.text_input("Enter Sentence 2", "")

if st.button("Check Similarity"):
    if sentence1 and sentence2:
        similarity_score = sentences_similar(sentence1, sentence2)
        st.write("Cosine Similarity:", similarity_score)
    else:
        st.warning("Please enter both sentences.")
