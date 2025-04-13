import streamlit as st
import pandas as pd
import pickle  # Changed from joblib to pickle
from PIL import Image
import gdown  

# ----------------------- DOWNLOAD MODEL -----------------------
# Replace <your_file_id> with your actual Google Drive file ID
url = 'https://drive.google.com/file/d/1QzhO53ttS18P6N8_5y2x3icatQZj3d8a/view?usp=sharing'
output = 'svd_model.pkl'

# Download model only if it doesn't exist locally
import os
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# ----------------------- LOAD DATA -----------------------
@st.cache_data
def load_data():
    anime = pd.read_csv("data/anime.csv")  # Must contain 'anime_id' and 'name'
    ratings = pd.read_csv("data/rating.csv")
    return anime, ratings

# ----------------------- LOAD MODEL -----------------------
@st.cache_resource
def load_model():
    # Using pickle to load the model
    with open("svd_model.pkl", 'rb') as f:
        model = pickle.load(f)  # Use pickle to load the model
    return model

# ----------------------- RECOMMENDATION FUNCTION -----------------------
def recommend_top_n(user_id, model, anime_df, rating_df, top_n=10):
    watched_ids = rating_df[rating_df['user_id'] == user_id]['anime_id'].unique()
    unwatched_ids = anime_df[~anime_df['anime_id'].isin(watched_ids)]['anime_id'].values

    input_df = pd.DataFrame({
        'user_id': [user_id] * len(unwatched_ids),
        'anime_id': unwatched_ids 
    })

    predicted_ratings = []
    for index, row in input_df.iterrows():
        predicted_rating = model.predict(row['user_id'], row['anime_id']).est
        predicted_ratings.append(predicted_rating)

    input_df['predicted_rating'] = predicted_ratings
    top_recs = input_df.sort_values(by='predicted_rating', ascending=False).head(top_n)
    merged = top_recs.merge(anime_df[['anime_id', 'name']], on='anime_id', how='left')
    return merged[['anime_id', 'name', 'predicted_rating']] 

# ----------------------- UI CONFIG -----------------------
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
tab1, tab2, tab3 = st.tabs(["Team Info", "Project Overview", "Recommender"])

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

# ----------------------- TAB 3: Recommender -----------------------
with tab3:
    st.header("Anime Recommender")
    st.write("Enter your user ID to get personalized anime recommendations.")

    anime_df, rating_df = load_data()
    model = load_model()

    user_id = st.number_input("Enter User ID:", min_value=1, step=1)

    if st.button("Get Recommendations"):
        if user_id not in rating_df['user_id'].unique():
            st.warning("User ID not found.")
        else:
            with st.spinner("Generating recommendations..."):
                recs = recommend_top_n(user_id, model, anime_df, rating_df)
                st.success("Here are your top picks!")
                st.dataframe(recs.reset_index(drop=True))
