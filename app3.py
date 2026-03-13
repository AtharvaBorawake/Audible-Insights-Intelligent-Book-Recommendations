import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="Audible Book Insights", page_icon="📚", layout="wide")

# Custom CSS for Branding and Text Contrast
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stButton>button { background-color: #f15a24; color: white; border-radius: 8px; font-weight: bold; }
    
    /* Book Card Styling */
    .book-card { 
        background-color: #f0f2f6 !important; 
        padding: 20px; 
        border-radius: 12px; 
        border-top: 5px solid #f15a24; 
        margin-bottom: 20px;
        color: #1a1a1a !important;
    }
    .book-card h3 { color: #f15a24 !important; margin-top: 0px; }
    .book-card p { color: #333333 !important; }
    
    /* FIX: Question Box Styling (Forced Dark Text) */
    .question-box { 
        background-color: #e9ecef !important; 
        color: #1a1a1a !important; /* Forces text to be dark/visible */
        padding: 15px; 
        border-radius: 8px; 
        margin-bottom: 10px; 
        border-left: 5px solid #f15a24;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_pickle("notebook/processed_books.pkl")
        with open("notebook/cosine_sim.pkl", "rb") as f:
            cosine_sim = pickle.load(f)
        with open("notebook/indices.pkl", "rb") as f:
            indices = pickle.load(f)
            
        raw_genres = df['Genre'].str.split().explode()
        clean_genres = raw_genres[
            (raw_genres.str.len() > 1) & 
            (~raw_genres.isin(['and', 'the', 'Books', 'In', 'Audible', 'Audiobooks', 'Originals']))
        ]
        return df, cosine_sim, indices, clean_genres
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None

df, cosine_sim, indices, clean_genres = load_data()

# --- RECOMMENDATION LOGIC ---
def get_recommendations(title, n=5):
    if title not in indices: return None
    
    idx = indices[title]
    if not isinstance(idx, (int, np.integer)): idx = idx[0]
        
    cluster_id = df.iloc[idx]["cluster"]
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    scored_books = []
    for i, sim in sim_scores:
        if i != idx and i < len(df) and df.iloc[i]["cluster"] == cluster_id:
            quality_score = (df.iloc[i]['weighted_score'] / 5.0)
            total_score = (sim * 0.6) + (quality_score * 0.4)
            scored_books.append((i, total_score, sim))
    
    return sorted(scored_books, key=lambda x: x[1], reverse=True)[:n]

def safe_rating_display(raw_rating):
    if pd.isna(raw_rating) or raw_rating < 0:
        return "Not Rated", ""
    return f"({raw_rating})", "⭐" * int(round(raw_rating))

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("assets/bookimage.jpg", width=250)
st.sidebar.markdown("### Navigation")
app_mode = st.sidebar.radio("Go to:", ["🔍 Recommend Engine", "📊 EDA & Project Q&A","👨‍💻 Developer Info"])

st.sidebar.divider()
st.sidebar.caption("Project: Audible Insights")

# ==========================================
# PAGE 1: RECOMMENDATION ENGINE
# ==========================================
if app_mode == "🔍 Recommend Engine" and df is not None:
    st.title("🎧 Audible Book Recommender")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Your Preferences")
        book_names = sorted(df['Book Name'].unique())
        selected_book = st.selectbox("Search for a Book:", book_names)
        num_recs = st.select_slider("Results:", options=[3, 5, 10], value=5)
        search_btn = st.button("Find Similar Books")

    with col2:
        if search_btn:
            results = get_recommendations(selected_book, n=num_recs)
            if results:
                source_row = df[df['Book Name'] == selected_book].iloc[0]
                source_genres = set(str(source_row['Genre']).split())
                
                st.subheader(f"Because you enjoyed '{selected_book}':")
                for i in range(0, len(results), 2):
                    cols = st.columns(2)
                    for j, (idx_rec, total_score, sim_val) in enumerate(results[i:i+2]):
                        row = df.iloc[idx_rec]
                        with cols[j]:
                            rating_text, stars = safe_rating_display(row['Rating'])
                            
                            st.markdown(f"""
                                <div class='book-card'>
                                    <h3>{row['Book Name']}</h3>
                                    <p><b>Author:</b> {row['Author']}</p>
                                    <p><b>Rating:</b> {stars} {rating_text}</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            rec_genres = set(str(row['Genre']).split())
                            common = source_genres.intersection(rec_genres)
                            if common:
                                st.caption(f"Tags: {', '.join(list(common)[:3])}")
                            st.progress(float(sim_val))
            else:
                st.info("No matching books found in this category.")

# ==========================================
# PAGE 2: IN-DEPTH EDA & PROJECT Q&A
# ==========================================
elif app_mode == "📊 EDA & Project Q&A" and df is not None:
    st.title("📊 Exploratory Data Analysis & Project Insights")
    

    # FIX: Grouped the content into 3 sub-tabs so it stays exactly where it belongs
    tab_viz, tab_qa, tab_scenario = st.tabs(["📈 Data Visualizations", "❓ General Q&A", "🎬 Scenario Applications"])

    # 1. VISUALIZATIONS TAB
    with tab_viz:
        st.header("Core Visualizations")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Most Popular Genres")
            st.bar_chart(clean_genres.value_counts().head(10))
            
            st.subheader("Ratings vs Review Counts")
            st.scatter_chart(df, x='Number of Reviews', y='Rating', color='#f15a24')
            
        with c2:
            st.subheader("Most Common Authors")
            st.bar_chart(df['Author'].value_counts().head(10))
            
            st.subheader("Feature Correlation")
            numeric_df = df[['Rating', 'Number of Reviews', 'Price', 'weighted_score']].dropna()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="Oranges", fmt=".2f", ax=ax)
            st.pyplot(fig)

    # 2. GENERAL Q&A TAB
    with tab_qa:
        st.header("Project Questions")
        
        with st.expander("🟢 Which authors have the highest-rated books?"):
            author_stats = df.groupby('Author').agg({'Rating':'mean', 'Book Name':'count'}).rename(columns={'Book Name':'Book Count'})
            top_rated = author_stats[author_stats['Book Count'] >= 3].sort_values('Rating', ascending=False).head(5)
            st.dataframe(top_rated)
            
        with st.expander("🟢 What is the average rating distribution across books?"):
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.histplot(df['Rating'].dropna(), bins=20, kde=True, color='#f15a24', ax=ax)
            st.pyplot(fig)

        with st.expander("🟡 Which combination of features provides the most accurate recommendations?"):
            st.markdown("""
            The **Quality-Hybrid Model** provided the highest Precision. It combined:
            1. **TF-IDF NLP Matrix:** Extracted features from descriptions.
            2. **K-Means Clustering:** Ensured books belonged to similar thematic groups.
            3. **Bayesian Weighted Rating:** Prevented low-popularity books from overpowering widely-loved books.
            """)

        with st.expander("🟡 What is the effect of author popularity on book ratings?"):
            author_pop = df.groupby('Author').agg({'Number of Reviews': 'sum', 'Rating': 'mean'}).dropna()
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.scatterplot(data=author_pop, x='Number of Reviews', y='Rating', alpha=0.6, color='teal', ax=ax)
            ax.set_xscale('log')
            st.pyplot(fig)
            st.caption("As author popularity increases, average ratings tend to stabilize around 4.3 - 4.7 stars.")

    # 3. SCENARIO APPLICATIONS TAB
    with tab_scenario:
        st.header("Scenario Based Applications")
        
        st.markdown("<div class='question-box'><b>Scenario 1:</b> A new user likes science fiction books. Which top 5 books should be recommended?</div>", unsafe_allow_html=True)
        sf_books = df[df['Genre'].str.contains('Science Fiction', case=False, na=False)]
        st.table(sf_books.sort_values('weighted_score', ascending=False)[['Book Name', 'Author', 'Rating']].head(5))

        st.markdown("<div class='question-box'><b>Scenario 2:</b> For a user who has previously rated thrillers highly, recommend similar books.</div>", unsafe_allow_html=True)
        thriller_books = df[df['Genre'].str.contains('Thriller', case=False, na=False)]
        st.table(thriller_books.sort_values('weighted_score', ascending=False)[['Book Name', 'Author', 'Rating']].head(5))

        st.markdown("<div class='question-box'><b>Scenario 3:</b> Identify books that are highly rated but have low popularity to recommend hidden gems.</div>", unsafe_allow_html=True)
        gems = df[(df['Rating'] >= 4.8) & (df['Number of Reviews'] > 0) & (df['Number of Reviews'] < df['Number of Reviews'].quantile(0.25))]
        st.table(gems[['Book Name', 'Author', 'Rating', 'Number of Reviews']].head(5))
# ==========================================
# PAGE 3: DEVELOPER INFO
# ==========================================
elif app_mode == "👨‍💻 Developer Info":
    st.title("👨‍💻 Developer Profile")
    
    col_img, col_info = st.columns([1, 2])
    
    with col_img:
        # You can replace this URL with your LinkedIn photo URL or a local file
        st.image("assets/atharvaimage.jpg", width=200)
    
    with col_info:
        st.markdown(f"""
        <div class='dev-card'>
            <h2>Atharva Borawake</h2>
            <p><b>Role:</b> Data Scientist / AI Engineer</p>
            <p><b>Education:</b> B.Tech / JSPM University</p>
            <p><b>Tech Stack :</b> Python, Machine Learning, Deep Learning, NLP, Streamlit</p>
            
            
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    st.subheader("Connect with Me")
    c1, c2, c3 = st.columns(3)
    c1.link_button("🌐 LinkedIn", "https://linkedin.com/in/atharva-borawake-76a370197")
    c2.link_button("💻 GitHub", "https://github.com/AtharvaBorawake")
    c3.link_button("📧 Email", "mailto:atharvaborawake210@gmail.com")