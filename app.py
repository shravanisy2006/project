import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    tags = pd.read_csv("tags.csv")
    links = pd.read_csv("links.csv")
    return movies, ratings, tags, links

movies, ratings, tags, links = load_data()

# -------------------- MERGE --------------------
df = ratings.merge(movies, on="movieId")
df = df.merge(tags, on=["userId","movieId"], how="left")
df['tag'] = df['tag'].fillna("")

# -------------------- FEATURES --------------------
avg_rating = ratings.groupby('movieId')['rating'].mean()
popularity = ratings.groupby('movieId')['rating'].count()

movies['avg_rating'] = movies['movieId'].map(avg_rating)
movies['popularity'] = movies['movieId'].map(popularity)

movies.fillna(0, inplace=True)

# -------------------- MATRIX --------------------
movie_matrix = ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)

# -------------------- CLUSTERING --------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(movie_matrix)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

movie_matrix['Cluster'] = clusters

# -------------------- SESSION STATE --------------------
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# -------------------- FUNCTIONS --------------------
def rate_movie(movie_id, rating):
    st.session_state.user_ratings[movie_id] = rating

def recommend(movie_id, genre=None):
    if movie_id not in movie_matrix.index:
        return pd.DataFrame()

    cluster = movie_matrix.loc[movie_id, 'Cluster']
    similar_movies = movie_matrix[movie_matrix['Cluster'] == cluster].index

    result = movies[movies['movieId'].isin(similar_movies)]

    if genre:
        result = result[result['genres'].str.contains(genre, case=False)]

    return result.sort_values(by='avg_rating', ascending=False).head(10)

def personalized():
    if not st.session_state.user_ratings:
        return pd.DataFrame()

    liked = [mid for mid, r in st.session_state.user_ratings.items() if r >= 4]
    if not liked:
        return pd.DataFrame()

    clusters = movie_matrix.loc[liked]['Cluster']
    rec = movie_matrix[movie_matrix['Cluster'].isin(clusters)].index

    result = movies[movies['movieId'].isin(rec)]
    return result.sort_values(by='avg_rating', ascending=False).head(10)

def genre_recommend(genre):
    return movies[movies['genres'].str.contains(genre, case=False)].sort_values(by='avg_rating', ascending=False).head(10)

def top_rated():
    return movies.sort_values(by='avg_rating', ascending=False).head(10)

def popular():
    return movies.sort_values(by='popularity', ascending=False).head(10)

# -------------------- DISPLAY GRID --------------------
def display_movies(df):

    cols = st.columns(5)

    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 5]:
            st.subheader(row['title'][:25])
            st.write("⭐", round(row['avg_rating'],2))

            rating = st.slider("Rate", 1, 5, key=f"rate{row['movieId']}")

            if st.button("Submit", key=f"btn{row['movieId']}"):
                rate_movie(row['movieId'], rating)
                st.success("Saved!")

# -------------------- UI --------------------

st.title("🎬 Smart Movie Recommendation System")

# Sidebar
st.sidebar.header("🔍 Controls")

movie_id = st.sidebar.number_input("Enter Movie ID", min_value=1, step=1)
genre = st.sidebar.selectbox("Select Genre", ["", "Action","Comedy","Drama","Romance","Thriller","Sci-Fi"])

if st.sidebar.button("Recommend"):

    st.subheader("🎯 Recommended Movies")
    recs = recommend(movie_id, genre)

    if recs.empty:
        st.warning("No recommendations found")
    else:
        display_movies(recs)

# -------------------- PERSONALIZED --------------------

st.subheader("🔥 Recommended For You")

pers = personalized()

if pers.empty:
    st.info("Rate some movies to unlock recommendations")
else:
    display_movies(pers)

# -------------------- CONTINUE WATCHING --------------------

st.subheader("▶️ Continue Watching")

liked = [mid for mid, r in st.session_state.user_ratings.items() if r >= 4]
cont = movies[movies['movieId'].isin(liked)]

if not cont.empty:
    display_movies(cont.head(10))

# -------------------- GENRE --------------------

st.subheader("🎭 Browse by Genre")

g = st.selectbox("Choose Genre", ["Action","Comedy","Drama","Romance"])

display_movies(genre_recommend(g))

# -------------------- TOP --------------------

st.subheader("⭐ Top Rated")
display_movies(top_rated())

# -------------------- POPULAR --------------------

st.subheader("🔥 Popular Movies")
display_movies(popular())
