import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import time
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Anime Recommendation System", layout="wide")

# ----------------------- CSS for banner -----------------------
st.markdown("""
    <style>
    .banner {
        background-color: #6c63ff;
        color: white;
        text-align: center;
        padding: 2rem;
        font-size: 2.5rem;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------- IMAGE BANNER -----------------------
banner_image = Image.open("Top-10-Best-Anime-Series-Of-All-Time-Ranked-1140x570.jpg")
st.image(banner_image, use_container_width=True)

# ----------------------- PAGE TITLE -----------------------
st.markdown("<h1 style='text-align: center; margin-top: -20px;'>Anime Recommendation System</h1>", unsafe_allow_html=True)

# ----------------------- TABS -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["Team Info", "Project Overview", "Exploratory Data Analysis", "Recommender"])

# ----------------------- TAB 1: Team Info -----------------------
with tab1:
    st.header("Team Info")
    st.write("**Team Members:**")
    st.write("- Sarah Mahlangu")
    st.write("- Busisiwe Mbewe")
    st.write("- Lusani Gumula")
    st.write("- Ammaarah Vaizie")

# ----------------------- TAB 2: Project Overview -----------------------
with tab2:
    st.header("Project Overview")
    st.write("""
     This project develops an unsupervised machine learning-based anime recommendation system that personalizes user experience, enhances content discovery, predicts user ratings, and showcases both collaborative and content-based recommendation techniques.
             """)
    st.write("**Key Features:**")
    st.markdown("- Detailed Exploratory Data Analysis that extensively looks into the relationships of the variables")
    st.markdown("""
        <div style='margin-left: 20px'>
        <p><strong>Multiple recommender models:</strong></p>
        <ul>
        <li>Content-Based Filtering using Cosine Similarity</li>
        <li>Collaborative Filtering using Matrix Factorization (SVD)</li>
        <li>Hybrid Approach: Weighted Combination of SVD and Content-Based Similarity</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("- User-friendly Streamlit interface")
    st.markdown("- Real-time recommendations from user-inputted anime content they have watched")

# ----------------------- TAB 3: Exploratory Data Analysis -----------------------
with tab3:
    st.header("EDA results")


# ----------------------- TAB 4: Prediction -----------------------
with tab4:
    st.header("Recommend Anime content")

    
