import streamlit as st
import pickle
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

# 1. Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Quora Question Similarity Finder")

# 2. User Input
q1 = st.text_input("Enter first question")
q2 = st.text_input("Enter second question")

if st.button("Predict"):
    # 3. Feature Engineering (Calculate everything first)
    # Basic features
    val_len1 = len(str(q1))
    val_len2 = len(str(q2))
    val_word1 = len(str(q1).split())
    val_word2 = len(str(q2).split())
    
    # Fuzzy features
    val_fuzz = fuzz.QRatio(q1, q2)
    val_partial = fuzz.partial_ratio(q1, q2)
    val_tset = fuzz.token_set_ratio(q1, q2)
    val_tsort = fuzz.token_sort_ratio(q1, q2)
    
    # 4. Create the DataFrame with exact column names from training
    cols = ['len_q1', 'len_q2', 'words_q1', 'words_q2', 'fuzz_score', 'fuzz_partial', 'token_set', 'token_sort']
    
    query_df = pd.DataFrame([[val_len1, val_len2, val_word1, val_word2, val_fuzz, val_partial, val_tset, val_tsort]], 
                            columns=cols)
    
    # 5. Prediction
    result = model.predict(query_df)[0]
    
    # 6. Display Result
    if result == 1:
        st.header("Duplicate")
        st.write("These questions have the same meaning.")
    else:
        st.header("Not Duplicate")
        st.write("These questions are different.")