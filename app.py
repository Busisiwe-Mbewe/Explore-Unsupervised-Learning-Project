import streamlit as st
st.title("Anime Recommender")

user_id = st.number_input("Enter your user ID:", min_value=1, step=1)
anime_name = st.selectbox("Select an anime:", anime['name'])
anime_id = anime[anime['name'] == anime_name]['anime_id'].values[0]

if st.button("Recommend"):
    recs = hybrid_recommendations(user_id, anime_id)
    st.subheader("Top Recommendations:")
    for r in recs:
        st.write(f"{r[1]} (Score: {r[2]:.2f})")