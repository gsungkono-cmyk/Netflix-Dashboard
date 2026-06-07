import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Netflix Dashboard",
    page_icon="🎬",
    layout="wide"
)

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("netflix_titles.csv")

# =====================
# SIDEBAR
# =====================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Page",
    [
        "Overview",
        "Visualizations",
        "Model Performance",
        "CRISP-DM"
    ]
)

# FILTERS
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Filters")

content_type = st.sidebar.selectbox(
    "Content Type",
    ["All", "Movie", "TV Show"]
)

if content_type != "All":
    df = df[df["type"] == content_type]

# =====================
# OVERVIEW
# =====================
if page == "Overview":

    st.title("🎬 Netflix Mature Content Prediction")
    st.info("""
    This dashboard analyzes Netflix titles and compares machine learning models
    for predicting mature content classification. Users can explore dataset insights,
    model performance, and the CRISP-DM process through an interactive interface.
    """)

    total_titles = len(df)
    total_movies = len(df[df["type"] == "Movie"])
    total_tvshows = len(df[df["type"] == "TV Show"])

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Titles", total_titles)
    c2.metric("Movies", total_movies)
    c3.metric("TV Shows", total_tvshows)


    st.write(df.head())

# =====================
# VISUALIZATION
# =====================
elif page == "Visualizations":

    st.title("📊 Netflix Dataset Exploration")

    st.subheader("1. Content Type Distribution")
    st.caption(
    "This chart compares the number of Movies and TV Shows available in the selected dataset."
    )

    fig, ax = plt.subplots()
    df["type"].value_counts().plot(kind="bar", ax=ax)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # =====================================
    # Release Year Distribution
    # =====================================

    st.subheader("2. Content Released by Year")
    st.caption(
    "This visualization shows how Netflix content production has changed over time."
    )

    fig, ax = plt.subplots(figsize=(10,4))

    df["release_year"].value_counts().sort_index().plot(
        ax=ax
    )

    ax.set_xlabel("Release Year")
    ax.set_ylabel("Number of Titles")

    st.pyplot(fig)

    # =====================================
    # Top Countries
    # =====================================

    st.subheader("3. Top 10 Countries")
    st.caption(
    "This chart highlights the countries contributing the largest amount of content to Netflix."
    )

    country_counts = (
        df["country"]
        .dropna()
        .str.split(", ")
        .explode()
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8,4))

    country_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Count")

    st.pyplot(fig)

    # =====================================
    # Top Genres
    # =====================================

    st.subheader("4. Top 10 Genres")
    st.caption(
    "This visualization identifies the most common genres found in the Netflix dataset."
    )

    genre_counts = (
        df["listed_in"]
        .str.split(", ")
        .explode()
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8,4))

    genre_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Count")

    st.pyplot(fig)

    # =====================================
    # Dataset Preview
    # =====================================

    st.subheader("5. Dataset Preview")
    st.caption(
    "A sample of the dataset used for analysis and model development."
    )

    st.dataframe(df.head(20))

# =====================
# MODEL PERFORMANCE
# =====================
elif page == "Model Performance":

    st.title("🤖 Model Performance")
    st.info("""
    This section compares the performance of three classification models:
    Logistic Regression, Random Forest, and Gradient Boosting.
    Evaluation metrics include Accuracy, F1-Score, and ROC-AUC.
    """)

    results = pd.DataFrame({
        "Model":[
            "Gradient Boosting",
            "Random Forest",
            "Logistic Regression"
        ],
        "Accuracy":[
            0.7081,
            0.7179,
            0.6990
        ],
        "F1 Score":[
            0.7174,
            0.7099,
            0.6843
        ],
        "ROC-AUC":[
            0.7938,
            0.8124,
            0.7844
        ]
    })

    st.dataframe(results)

    st.subheader("F1 Score Comparison")
    st.bar_chart(
        results.set_index("Model")["F1 Score"]
    )

    st.success(
        "Best Model: Gradient Boosting (F1 Score = 0.7174)"
    )

# =====================
# CRISP DM
# =====================
else:

        st.title("📚 CRISP-DM Process")

        st.header("1️⃣ Business Understanding")

        st.write("""
        The objective of this project is to predict whether a Netflix title
        belongs to the Mature Content category or Non-Mature Content category.
        This prediction can help streaming platforms better understand content
        classification and improve recommendation systems.
        """)
        
        st.header("2️⃣ Data Understanding")
        
        st.write("""
        The dataset used is the Netflix Titles Dataset containing 8,807 records.
        
        Main attributes include:
        - Type (Movie / TV Show)
        - Release Year
        - Country
        - Genre
        - Cast
        - Director
        - Description
        - Duration
        """)
        
        st.header("3️⃣ Data Preparation")
        
        st.write("""
        Several preprocessing steps were performed:
        
        ✔ Handling missing values
        
        ✔ Feature engineering
        
        ✔ Creating new variables:
        - genre_count
        - country_count
        - cast_count
        - description_length
        - content_age_when_added
        
        ✔ Data transformation and encoding
        """)
        
        st.header("4️⃣ Modeling")
        
        st.write("""
        Three machine learning algorithms were tested:
        
        • Logistic Regression
        
        • Random Forest
        
        • Gradient Boosting
        """)
        
        st.header("5️⃣ Evaluation")
        
        results = pd.DataFrame({
            "Model": [
                "Gradient Boosting",
                "Random Forest",
                "Logistic Regression"
            ],
            "Accuracy": [0.7081, 0.7179, 0.6990],
            "F1 Score": [0.7174, 0.7099, 0.6843],
            "ROC-AUC": [0.7938, 0.8124, 0.7844]
        })
        
        st.dataframe(results)
        
        st.success(
            "Gradient Boosting achieved the highest F1 Score (0.7174), making it the best-performing model for this project."
        )
        
        st.header("6️⃣ Deployment")
        
        st.write("""
        The final solution was deployed using Streamlit,
        allowing users to explore the dataset, visualize insights,
        and review machine learning results through an interactive dashboard.
        """)
