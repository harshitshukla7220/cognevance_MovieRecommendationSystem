
import pickle
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(layout="wide")
st.header("Movie Recommender System Using Machine Learning")


def fetch_poster(movie_id):
    """Fetch a movie poster from TMDB."""
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        "?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")

        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path

    except requests.exceptions.RequestException:
        pass

    return "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"


# Load model files from the same folder as app.py
try:
    with open(BASE_DIR / "movie_dict.pkl", "rb") as file:
        movies_dict = pickle.load(file)

    movies = pd.DataFrame(movies_dict)

    with open(BASE_DIR / "similarity.pkl", "rb") as file:
        similarity = pickle.load(file)

except FileNotFoundError as error:
    st.error(f"Model file not found: {error.filename}")
    st.stop()

except (pickle.UnpicklingError, EOFError, ValueError) as error:
    st.error(f"Could not load the model files: {error}")
    st.stop()


def recommend(movie):
    """Recommend five similar movies."""
    matches = movies.index[movies["title"] == movie].tolist()

    if not matches:
        st.error("Movie not found in the dataset.")
        return [], [], [], []

    index = matches[0]
    distances = sorted(
        enumerate(similarity[index]),
        key=lambda item: item[1],
        reverse=True,
    )

    names = []
    posters = []
    years = []
    ratings = []

    for movie_index, _score in distances[1:6]:
        movie_row = movies.iloc[movie_index]
        names.append(movie_row["title"])
        posters.append(fetch_poster(movie_row["movie_id"]))
        years.append(movie_row.get("year"))
        ratings.append(movie_row.get("vote_average", 0))

    return names, posters, years, ratings


movie_list = movies["title"].dropna().tolist()

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list,
)

if st.button("Show Recommendation"):
    with st.spinner("Finding recommendations..."):
        names, posters, years, ratings = recommend(selected_movie)

    if names:
        cols = st.columns(len(names))

        for i, col in enumerate(cols):
            with col:
                st.write(names[i])
                st.image(posters[i], use_container_width=True)

                year = years[i]
                if pd.notna(year):
                    st.caption(f"Year: {int(year)}")
                else:
                    st.caption("Year: N/A")

                rating = ratings[i]
                if pd.notna(rating):
                    st.caption(f"Rating: {float(rating):.1f} ⭐")
                else:
                    st.caption("Rating: N/A")
