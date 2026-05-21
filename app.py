import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Netflix Recommendation Engine",
    page_icon="🎬",
    layout="wide"
)


# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/netflix_ratings.csv")
    df = df.drop_duplicates()

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["MovieID"] = df["MovieID"].astype(int)
    df["Rating"] = df["Rating"].astype(int)
    df["Year"] = df["Year"].astype(int)

    return df


df = load_data()


# -----------------------------
# Prepare movie stats
# -----------------------------
@st.cache_data
def prepare_movie_stats(df):
    movie_stats = df.groupby(["MovieID", "Title", "Genre", "Year"]).agg(
        avg_rating=("Rating", "mean"),
        rating_count=("Rating", "count")
    ).reset_index()

    movie_stats["popularity_score"] = (
        movie_stats["avg_rating"] * np.log1p(movie_stats["rating_count"])
    )

    return movie_stats


movie_stats = prepare_movie_stats(df)
@st.cache_resource
def train_rating_prediction_model(df):
    model_df = df.copy()

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    genre_encoder = LabelEncoder()

    model_df["user_encoded"] = user_encoder.fit_transform(model_df["CustomerID"])
    model_df["movie_encoded"] = movie_encoder.fit_transform(model_df["MovieID"])
    model_df["genre_encoded"] = genre_encoder.fit_transform(model_df["Genre"])

    X = model_df[["user_encoded", "movie_encoded", "genre_encoded", "Year"]]
    y = model_df["Rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    return rf_model, rmse, mae, user_encoder, movie_encoder, genre_encoder


rf_model, rmse, mae, user_encoder, movie_encoder, genre_encoder = train_rating_prediction_model(df)

# -----------------------------
# Prepare content similarity
# -----------------------------
@st.cache_data
def prepare_content_model(df):
    movies = df[["MovieID", "Title", "Genre", "Year"]].drop_duplicates().reset_index(drop=True)

    movies["features"] = (
        movies["Title"].fillna("") + " " +
        movies["Genre"].fillna("") + " " +
        movies["Year"].astype(str)
    )

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["features"])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return movies, cosine_sim


movies, cosine_sim = prepare_content_model(df)


# -----------------------------
# Recommendation functions
# -----------------------------
def get_popular_movies(genre="All", top_n=10):
    data = movie_stats.copy()

    if genre != "All":
        data = data[data["Genre"] == genre]

    return data.sort_values("popularity_score", ascending=False).head(top_n)


def recommend_similar_movies(movie_title, top_n=10):
    movie_title = movie_title.lower()

    title_list = movies["Title"].str.lower().tolist()

    if movie_title not in title_list:
        return pd.DataFrame()

    index = title_list.index(movie_title)

    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:top_n + 1]

    movie_indices = [i[0] for i in similarity_scores]

    result = movies.iloc[movie_indices][["MovieID", "Title", "Genre", "Year"]].copy()
    result["Similarity Score"] = [round(i[1], 3) for i in similarity_scores]

    return result


def recommend_for_user(user_id, top_n=10, genre_filter="All"):
    user_data = df[df["CustomerID"] == user_id]

    if user_data.empty:
        return pd.DataFrame(), pd.DataFrame()

    watched_movies = user_data["MovieID"].unique()

    favorite_genres = user_data.groupby("Genre")["Rating"].mean().sort_values(ascending=False)

    candidate_movies = movie_stats[~movie_stats["MovieID"].isin(watched_movies)].copy()

    if genre_filter != "All":
        candidate_movies = candidate_movies[candidate_movies["Genre"] == genre_filter]

    candidate_movies["genre_preference_score"] = candidate_movies["Genre"].map(favorite_genres).fillna(3)

    candidate_movies["hybrid_score"] = (
        0.45 * candidate_movies["avg_rating"] +
        0.35 * candidate_movies["genre_preference_score"] +
        0.20 * (
            candidate_movies["popularity_score"] / candidate_movies["popularity_score"].max()
        )
    )

    recommendations = candidate_movies.sort_values("hybrid_score", ascending=False).head(top_n)

    return recommendations, favorite_genres.reset_index()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎬 Netflix ML App")

menu = st.sidebar.radio(
    "Choose Page",
    [
        "🏠 Home",
        "📊 EDA Dashboard",
        "🔥 Popular Movies",
        "🎞️ Movie Recommendation",
        "👤 User Recommendation",
        "📈 Model Evaluation",
        "🧠 Project Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("Built using Python, Pandas, Scikit-learn, and Streamlit")


# -----------------------------
# Home Page
# -----------------------------
if menu == "🏠 Home":
    st.title("🎬 Netflix Movie Recommendation Engine")
    st.write(
        """
        This is an interactive machine learning project that recommends movies
        based on popularity, movie similarity, genre preference, and user behavior.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Ratings", len(df))
    col2.metric("Unique Users", df["CustomerID"].nunique())
    col3.metric("Unique Movies", df["MovieID"].nunique())
    col4.metric("Total Genres", df["Genre"].nunique())

    st.markdown("---")

    st.subheader("🚀 What this app can do")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Recommendation Features")
        st.write("✅ Popular movie recommendations")
        st.write("✅ Movie-based similar recommendations")
        st.write("✅ User-based personalized recommendations")
        st.write("✅ Genre filtering")
        st.write("✅ Hybrid scoring system")

    with col2:
        st.write("### Skills Shown")
        st.write("✅ Python")
        st.write("✅ Pandas")
        st.write("✅ Machine Learning")
        st.write("✅ Recommendation Systems")
        st.write("✅ Streamlit Deployment")


# -----------------------------
# EDA Dashboard
# -----------------------------
elif menu == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rating Distribution")
        rating_counts = df["Rating"].value_counts().sort_index()
        st.bar_chart(rating_counts)

    with col2:
        st.subheader("Average Rating by Genre")
        genre_rating = df.groupby("Genre")["Rating"].mean().sort_values(ascending=False)
        st.bar_chart(genre_rating)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Number of Ratings by Genre")
        genre_count = df["Genre"].value_counts()
        st.bar_chart(genre_count)

    with col4:
        st.subheader("Movies Released by Year")
        year_count = df[["MovieID", "Year"]].drop_duplicates()["Year"].value_counts().sort_index()
        st.line_chart(year_count)

    st.markdown("---")

    st.subheader("Genre Summary Table")

    genre_summary = df.groupby("Genre").agg(
        Average_Rating=("Rating", "mean"),
        Total_Ratings=("Rating", "count"),
        Unique_Movies=("MovieID", "nunique")
    ).reset_index().sort_values("Average_Rating", ascending=False)

    st.dataframe(genre_summary, use_container_width=True)


# -----------------------------
# Popular Movies Page
# -----------------------------
elif menu == "🔥 Popular Movies":
    st.title("🔥 Popular Movies Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        genres = ["All"] + sorted(df["Genre"].unique().tolist())
        selected_genre = st.selectbox("Select Genre", genres)

    with col2:
        top_n = st.slider("Number of movies", 5, 30, 10)

    popular_movies = get_popular_movies(selected_genre, top_n)

    st.subheader("Top Recommended Popular Movies")
    st.dataframe(
        popular_movies[
            ["Title", "Genre", "Year", "avg_rating", "rating_count", "popularity_score"]
        ],
        use_container_width=True
    )

    st.write(
        """
        These movies are recommended based on a popularity score calculated using
        average rating and total number of ratings.
        """
    )


# -----------------------------
# Movie Recommendation Page
# -----------------------------
elif menu == "🎞️ Movie Recommendation":
    st.title("🎞️ Movie-Based Recommendation")

    st.write(
        """
        Select a movie and the system will recommend similar movies using
        content-based filtering with TF-IDF and cosine similarity.
        """
    )

    selected_movie = st.selectbox(
        "Search or select a movie",
        sorted(movies["Title"].unique())
    )

    top_n = st.slider("How many similar movies do you want?", 5, 20, 10)

    if st.button("Recommend Similar Movies"):
        recommendations = recommend_similar_movies(selected_movie, top_n)

        if recommendations.empty:
            st.error("Movie not found.")
        else:
            st.success(f"Movies similar to: {selected_movie}")
            st.dataframe(recommendations, use_container_width=True)

            selected_info = movies[movies["Title"] == selected_movie].iloc[0]

            st.markdown("---")
            st.subheader("Why these movies?")
            st.write(
                f"""
                The selected movie belongs to **{selected_info['Genre']}** genre.
                The recommendation system compares the selected movie with all other
                movies using title, genre, and year-based text similarity.
                """
            )


# -----------------------------
# User Recommendation Page
# -----------------------------
elif menu == "👤 User Recommendation":
    st.title("👤 Personalized User Recommendation")

    st.write(
        """
        Enter a customer ID and the system will recommend movies based on that user's
        previous ratings and genre preferences.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        user_id = st.number_input(
            "Enter Customer ID",
            min_value=int(df["CustomerID"].min()),
            max_value=int(df["CustomerID"].max()),
            value=int(df["CustomerID"].min())
        )

    with col2:
        genres = ["All"] + sorted(df["Genre"].unique().tolist())
        genre_filter = st.selectbox("Filter by Genre", genres)

    with col3:
        top_n = st.slider("Number of recommendations", 5, 20, 10)

    user_history = df[df["CustomerID"] == user_id]

    st.markdown("---")

    col4, col5 = st.columns(2)

    with col4:
        st.subheader("User Rating History")
        st.dataframe(
            user_history[["Title", "Genre", "Year", "Rating"]].sort_values("Rating", ascending=False),
            use_container_width=True
        )

    with col5:
        st.subheader("User Favorite Genres")
        fav_genres = user_history.groupby("Genre")["Rating"].mean().sort_values(ascending=False)
        st.bar_chart(fav_genres)

    if st.button("Generate Personalized Recommendations"):
        recommendations, favorite_genres = recommend_for_user(user_id, top_n, genre_filter)

        if recommendations.empty:
            st.error("No recommendations found for this user.")
        else:
            st.success(f"Top {top_n} recommendations for Customer ID {user_id}")

            st.dataframe(
                recommendations[
                    [
                        "Title",
                        "Genre",
                        "Year",
                        "avg_rating",
                        "rating_count",
                        "genre_preference_score",
                        "hybrid_score"
                    ]
                ],
                use_container_width=True
            )

            st.markdown("---")
            st.subheader("Recommendation Explanation")
            st.write(
                """
                These movies are recommended using a hybrid score based on:
                - Movie average rating
                - User's preferred genres
                - Movie popularity score
                """
            )


# -----------------------------
# Project Insights Page
# -----------------------------
elif menu == "📈 Model Evaluation":
    st.title("📈 Model Evaluation")

    st.write(
        """
        This page evaluates a machine learning model that predicts movie ratings
        using user ID, movie ID, genre, and release year.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Used", "Random Forest Regressor")
    col2.metric("RMSE", round(rmse, 3))
    col3.metric("MAE", round(mae, 3))

    st.markdown("---")

    st.subheader("What do these metrics mean?")

    st.write(
        """
        **RMSE** means Root Mean Squared Error.  
        It tells how far the predicted rating is from the actual rating.

        **MAE** means Mean Absolute Error.  
        It tells the average rating prediction mistake.

        Lower RMSE and MAE values mean the model is performing better.
        """
    )

    st.markdown("---")

    st.subheader("Try Rating Prediction")

    col4, col5 = st.columns(2)

    with col4:
        selected_user = st.selectbox(
            "Select Customer ID",
            sorted(df["CustomerID"].unique())
        )

    with col5:
        selected_movie = st.selectbox(
            "Select Movie",
            sorted(df["Title"].unique())
        )

    movie_row = df[df["Title"] == selected_movie].iloc[0]

    if st.button("Predict Rating"):
        try:
            user_encoded = user_encoder.transform([selected_user])[0]
            movie_encoded = movie_encoder.transform([movie_row["MovieID"]])[0]
            genre_encoded = genre_encoder.transform([movie_row["Genre"]])[0]

            input_data = pd.DataFrame(
                [[user_encoded, movie_encoded, genre_encoded, movie_row["Year"]]],
                columns=["user_encoded", "movie_encoded", "genre_encoded", "Year"]
            )

            predicted_rating = rf_model.predict(input_data)[0]

            st.success(
                f"Predicted rating for Customer {selected_user} on '{selected_movie}' is: {round(predicted_rating, 2)} ⭐"
            )

            st.write("### Movie Details")
            st.write("Title:", movie_row["Title"])
            st.write("Genre:", movie_row["Genre"])
            st.write("Year:", movie_row["Year"])

        except Exception as e:
            st.error("Prediction failed.")
            st.write(e)

    st.markdown("---")

    # st.subheader("Why this page is important for resume?")

    st.write(
        """
        This page shows that the project is not only a dashboard.
        It includes a real machine learning model, train-test split,
        prediction, and evaluation using RMSE and MAE.
        """
    )
elif menu == "🧠 Project Insights":
    st.title("🧠 Project Insights")

    best_genre = df.groupby("Genre")["Rating"].mean().idxmax()
    worst_genre = df.groupby("Genre")["Rating"].mean().idxmin()
    most_rated_genre = df["Genre"].value_counts().idxmax()

    col1, col2, col3 = st.columns(3)

    col1.metric("Best Rated Genre", best_genre)
    col2.metric("Lowest Rated Genre", worst_genre)
    col3.metric("Most Rated Genre", most_rated_genre)

    st.markdown("---")

    st.subheader("Business Understanding")

    st.write(
        """
        A recommendation engine helps OTT platforms increase user engagement by
        suggesting movies that match user interests. This project uses rating history,
        genre behavior, and movie popularity to generate useful recommendations.
        """
    )

    st.subheader("Machine Learning Techniques Used")

    st.write(
        """
        1. Popularity-based recommendation  
        2. Content-based filtering using TF-IDF and cosine similarity  
        3. User preference-based recommendation  
        4. Hybrid recommendation scoring  
        """
    )

    # st.subheader("Resume Value")

    st.write(
        """
        This project demonstrates skills in data cleaning, exploratory data analysis,
        recommendation systems, machine learning, dashboard development, and deployment.
        """
    )
