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

# -------------------- PREPARE DATA --------------------
df = ratings.merge(movies, on="movieId")
df = df.merge(tags, on=["userId","movieId"], how="left")
df['tag'] = df['tag'].fillna("")

# stats
avg_rating = ratings.groupby('movieId')['rating'].mean()
popularity = ratings.groupby('movieId')['rating'].count()

movies['avg_rating'] = movies['movieId'].map(avg_rating)
movies['popularity'] = movies['movieId'].map(popularity)
movies.fillna(0, inplace=True)

# matrix
movie_matrix = ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)

# clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(movie_matrix)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

movie_matrix['Cluster'] = clusters

# -------------------- SESSION --------------------
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# -------------------- FUNCTIONS --------------------
def recommend(movie_id):
    if movie_id not in movie_matrix.index:
        return pd.DataFrame()
    cluster = movie_matrix.loc[movie_id, 'Cluster']
    similar_movies = movie_matrix[movie_matrix['Cluster'] == cluster].index
    return movies[movies['movieId'].isin(similar_movies)].sort_values(by='avg_rating', ascending=False).head(10)

def personalized():
    liked = [mid for mid, r in st.session_state.user_ratings.items() if r >= 4]
    if not liked:
        return pd.DataFrame()
    clusters = movie_matrix.loc[liked]['Cluster']
    rec = movie_matrix[movie_matrix['Cluster'].isin(clusters)].index
    return movies[movies['movieId'].isin(rec)].sort_values(by='avg_rating', ascending=False).head(10)

def genre_recommend(genre):
    return movies[movies['genres'].str.contains(genre, case=False)].sort_values(by='avg_rating', ascending=False).head(10)

def top_rated():
    return movies.sort_values(by='avg_rating', ascending=False).head(10)

def popular():
    return movies.sort_values(by='popularity', ascending=False).head(10)

# -------------------- DISPLAY --------------------
def display_movies(df, section):

    for i in range(0, len(df), 5):
        row_movies = df.iloc[i:i+5]
        cols = st.columns(5)

        for j, (_, movie) in enumerate(row_movies.iterrows()):
            with cols[j]:
                st.markdown(f"### 🎬 {movie['title'][:25]}")
                st.write(f"⭐ {round(movie['avg_rating'],2)}")

                if st.button("👍 Like", key=f"{section}_like_{movie['movieId']}"):
                    st.session_state.user_ratings[movie['movieId']] = 5
                    st.success("Added")

                if st.button("⭐ Rate", key=f"{section}_ratebtn_{movie['movieId']}"):
                    rating = st.slider(
                        "Your Rating",
                        1, 5,
                        key=f"{section}_slider_{movie['movieId']}"
                    )
                    if st.button("Save", key=f"{section}_save_{movie['movieId']}"):
                        st.session_state.user_ratings[movie['movieId']] = rating
                        st.success("Saved")

# -------------------- UI --------------------
st.set_page_config(page_title="Movie App", layout="wide")

st.title("🍿 Movie App")
st.write("Find movies you'll love ❤️")

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Home",
    "🎯 Find Movies",
    "🔥 Popular",
    "⭐ Top Rated",
    "🎭 Genre",
    "👤 My Space"
])

# -------------------- HOME --------------------
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔥 Popular"):
            display_movies(popular(), "home_pop")

    with col2:
        if st.button("⭐ Top Rated"):
            display_movies(top_rated(), "home_top")

    with col3:
        if st.button("🎲 Surprise Me"):
            display_movies(movies.sample(5), "random")

# -------------------- FIND MOVIES --------------------
with tab2:
    st.title("🎯 Find Similar Movies")

    movie_id = st.number_input("Enter Movie ID", min_value=1, step=1)

    if st.button("Show Suggestions"):
        recs = recommend(movie_id)

        if recs.empty:
            st.info("No movies found")
        else:
            display_movies(recs, "recommend")

# -------------------- POPULAR --------------------
with tab3:
    st.title("🔥 Popular")

    if st.button("Show Popular Movies"):
        display_movies(popular(), "popular")

# -------------------- TOP RATED --------------------
with tab4:
    st.title("⭐ Top Rated")

    if st.button("Show Top Movies"):
        display_movies(top_rated(), "top")

# -------------------- GENRE --------------------
with tab5:
    st.title("🎭 Browse by Genre")

    g = st.selectbox("Choose Genre", ["Action","Comedy","Drama","Romance"])

    if st.button("Show Movies"):
        display_movies(genre_recommend(g), "genre")

# -------------------- MY SPACE --------------------
with tab6:
    st.title("👤 My Space")

    st.subheader("Continue Watching")

    liked = [mid for mid, r in st.session_state.user_ratings.items() if r >= 4]
    cont = movies[movies['movieId'].isin(liked)]

    if not cont.empty:
        display_movies(cont, "continue")
    else:
        st.write("Start liking movies")

    st.subheader("Recommended For You")

    pers = personalized()

    if not pers.empty:
        display_movies(pers, "personalized")
    else:
        st.write("Rate movies to unlock suggestions")
