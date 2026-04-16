import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    tags = pd.read_csv("tags.csv")
    links = pd.read_csv("links.csv")
    return movies, ratings, tags, links

movies, ratings, tags, links = load_data()

# ---------------------- MERGE DATA ----------------------
df = ratings.merge(movies, on="movieId")
df = df.merge(tags, on=["userId", "movieId"], how="left")
df['tag'] = df['tag'].fillna("")

# ---------------------- FEATURE ENGINEERING ----------------------
avg_rating = ratings.groupby('movieId')['rating'].mean()
popularity = ratings.groupby('movieId')['rating'].count()

movies['avg_rating'] = movies['movieId'].map(avg_rating)
movies['popularity'] = movies['movieId'].map(popularity)

movies.fillna(0, inplace=True)

# ---------------------- MATRIX ----------------------
movie_matrix = ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)

# ---------------------- SCALING ----------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(movie_matrix)

# ---------------------- KMEANS ----------------------
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

movie_matrix['Cluster'] = clusters

# ---------------------- FUNCTIONS ----------------------

def recommend(movie_id, genre=None):
    if movie_id not in movie_matrix.index:
        return pd.DataFrame()

    cluster = movie_matrix.loc[movie_id, 'Cluster']
    similar_movies = movie_matrix[movie_matrix['Cluster'] == cluster].index

    result = movies[movies['movieId'].isin(similar_movies)]

    if genre:
        result = result[result['genres'].str.contains(genre, case=False)]

    return result.sort_values(by='avg_rating', ascending=False).head(10)


def top_rated():
    return movies.sort_values(by='avg_rating', ascending=False).head(10)


def popular_movies():
    return movies.sort_values(by='popularity', ascending=False).head(10)


def genre_recommend(genre):
    return movies[movies['genres'].str.contains(genre, case=False)].sort_values(by='avg_rating', ascending=False).head(10)


# ---------------------- UI ----------------------

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.markdown(
    """
    <style>
    .title {
        font-size:40px;
        font-weight:bold;
        color:#FF4B4B;
    }
    .card {
        padding:15px;
        border-radius:15px;
        background-color:#1e1e1e;
        margin-bottom:10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🎬 Smart Movie Recommendation System</div>', unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------

st.sidebar.header("🔍 Filters")

movie_id = st.sidebar.number_input("Enter Movie ID", min_value=1, step=1)

genre = st.sidebar.selectbox("Select Genre", ["", "Action", "Comedy", "Drama", "Romance", "Thriller", "Sci-Fi"])

# ---------------------- BUTTON ----------------------

if st.sidebar.button("🎯 Recommend"):

    st.subheader("🎯 Recommended Movies")

    recs = recommend(movie_id, genre)

    if recs.empty:
        st.warning("No recommendations found")
    else:
        for _, row in recs.iterrows():
            st.markdown(f"""
            <div class="card">
            <b>{row['title']}</b><br>
            ⭐ Rating: {round(row['avg_rating'],2)}<br>
            🎭 Genre: {row['genres']}
            </div>
            """, unsafe_allow_html=True)


# ---------------------- GENRE SECTION ----------------------

st.subheader("🎭 Genre-Based Recommendations")

selected_genre = st.selectbox("Choose Genre", ["Action", "Comedy", "Drama", "Romance"])

genre_movies = genre_recommend(selected_genre)

for _, row in genre_movies.iterrows():
    st.markdown(f"""
    <div class="card">
    <b>{row['title']}</b><br>
    ⭐ Rating: {round(row['avg_rating'],2)}
    </div>
    """, unsafe_allow_html=True)

# ---------------------- TOP RATED ----------------------

st.subheader("⭐ Top Rated Movies")

top = top_rated()

for _, row in top.iterrows():
    st.markdown(f"""
    <div class="card">
    <b>{row['title']}</b><br>
    ⭐ {round(row['avg_rating'],2)}
    </div>
    """, unsafe_allow_html=True)

# ---------------------- POPULAR ----------------------

st.subheader("🔥 Popular Movies")

pop = popular_movies()

for _, row in pop.iterrows():
    st.markdown(f"""
    <div class="card">
    <b>{row['title']}</b><br>
    👥 {int(row['popularity'])} ratings
    </div>
    """, unsafe_allow_html=True)