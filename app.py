import streamlit as st
import pandas as pd
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

# -------------------- SESSION --------------------
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
def display_movies(df, section):

    cols = st.columns(5)

    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 5]:
            st.subheader(row['title'][:25])
            st.write("⭐", round(row['avg_rating'],2))

            rating = st.slider(
                "Rate", 1, 5,
                key=f"{section}_rate_{row['movieId']}"
            )

            if st.button(
                "Submit",
                key=f"{section}_btn_{row['movieId']}"
            ):
                rate_movie(row['movieId'], rating)
                st.success("Saved!")

# -------------------- UI --------------------

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Smart Movie Recommendation System")

# -------------------- TABS --------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Home",
    "🎯 Recommend",
    "🔥 Popular",
    "⭐ Top Rated",
    "🎭 Genre",
    "👤 My Space"
])

# -------------------- HOME --------------------

with tab1:
    st.header("Welcome 🎉")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔥 Popular Movies"):
            display_movies(popular(), "home_pop")

    with col2:
        if st.button("⭐ Top Rated"):
            display_movies(top_rated(), "home_top")

    with col3:
        if st.button("🎯 Recommend Movies"):
            st.info("Go to Recommend tab")

    st.subheader("🎲 Surprise Me")

    if st.button("Give Random Movie"):
        random_movie = movies.sample(5)
        display_movies(random_movie, "random")

# -------------------- RECOMMEND --------------------

with tab2:
    st.header("🎯 Recommendation")

    movie_id = st.number_input("Enter Movie ID", min_value=1, step=1)
    genre = st.selectbox("Genre Filter", ["", "Action","Comedy","Drama","Romance"])

    if st.button("Recommend Now"):
        recs = recommend(movie_id, genre)

        if recs.empty:
            st.warning("No recommendations found")
        else:
            display_movies(recs, "recommend")

# -------------------- POPULAR --------------------

with tab3:
    st.header("🔥 Popular Movies")

    if st.button("Show Popular"):
        display_movies(popular(), "popular")

# -------------------- TOP --------------------

with tab4:
    st.header("⭐ Top Rated")

    if st.button("Show Top Rated"):
        display_movies(top_rated(), "top")

# -------------------- GENRE --------------------

with tab5:
    st.header("🎭 Genre")

    g = st.selectbox("Choose Genre", ["Action","Comedy","Drama","Romance"])

    if st.button("Show Genre Movies"):
        display_movies(genre_recommend(g), "genre")

# -------------------- MY SPACE --------------------

with tab6:
    st.header("👤 My Space")

    st.subheader("▶️ Continue Watching")

    liked = [mid for mid, r in st.session_state.user_ratings.items() if r >= 4]
    cont = movies[movies['movieId'].isin(liked)]

    if not cont.empty:
        display_movies(cont, "continue")
    else:
        st.info("No liked movies yet")

    st.subheader("🔥 Recommended For You")

    pers = personalized()

    if not pers.empty:
        display_movies(pers, "personalized")
    else:
        st.info("Rate movies to unlock recommendations")
