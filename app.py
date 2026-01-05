import streamlit as st
import pickle
import pandas as pd
import os
import numpy as np
from fuzzywuzzy import fuzz


st.set_page_config(page_title="Quora Duplicate Detector")
st.title("Quora Question Similarity Finder")


model_path = os.path.join(os.getcwd(), 'model.pkl')


model = None

if not os.path.exists(model_path):
    st.error(f"🚨 Error: 'model.pkl' not found at {model_path}. Please check your GitHub files!")
else:
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")


q1 = st.text_input("Enter first question")
q2 = st.text_input("Enter second question")

if st.button("Predict"):
    if model is None:
        st.error("Cannot predict because the model is not loaded.")
    elif not q1 or not q2:
        st.warning("Please enter both questions first.")
    else:
        
        val_len1 = len(str(q1))
        val_len2 = len(str(q2))
        val_word1 = len(str(q1).split())
        val_word2 = len(str(q2).split())
        
        val_fuzz = fuzz.QRatio(q1, q2)
        val_partial = fuzz.partial_ratio(q1, q2)
        val_tset = fuzz.token_set_ratio(q1, q2)
        val_tsort = fuzz.token_sort_ratio(q1, q2)
        
        
        cols = ['len_q1', 'len_q2', 'words_q1', 'words_q2', 'fuzz_score', 'fuzz_partial', 'token_set', 'token_sort']
        query_df = pd.DataFrame([[val_len1, val_len2, val_word1, val_word2, val_fuzz, val_partial, val_tset, val_tsort]], 
                                columns=cols)
        
       
        result = model.predict(query_df)[0]
        
        if result == 1:
            st.header("Duplicate")
            st.write("These questions share the same intent.")
        else:
            st.header("Not Duplicate")
            st.write("These questions are different.")
