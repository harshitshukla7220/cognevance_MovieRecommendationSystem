# cognevance_MovieRecommendationSystem


## Project Overview
A content-based movies recommendation system developed as part of the Cognevance Technologies AI/ML project.

The application recommends five similar movies based on the selected movie's metadata.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- NLTK
- Streamlit
- TMDB API

## Dataset
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Dataset source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

## Workflow
1. Load and merge movie datasets.
2. Clean and preprocess movie metadata.
3. Combine movie overview, genres, keywords, cast, and director into text tags.
4. Apply stemming and CountVectorizer.
5. Calculate cosine similarity.
6. Save the processed movie data and similarity matrix.
7. Display movie recommendations through Streamlit.

## Run the Project

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Model Files
- `movie_dict.pkl`: Processed movie information.
- `similarity.pkl`: Movie similarity matrix.

## Output
The application displays five recommended movies with posters, release years, and ratings when available.

## Author
Harshit Shukla
