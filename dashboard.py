import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# --- Configuration & Data Loading ---
load_dotenv()
st.set_page_config(page_title="Semantic Book Recommender", layout="wide")

@st.cache_resource
def init_db():
    books = pd.read_csv("books_with_emotions.csv")
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"],
    )

    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(documents, OpenAIEmbeddings())
    return books, db_books

books, db_books = init_db()

# --- Logic ---
def retrieve_semantic_recommendations(query, category="All", tone="All", initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    books_list = [int(rec[0].page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].copy()

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]
    
    tone_map = {
        "Happy": "joy_x",
        "Surprising": "surprise_x",
        "Angry": "anger_x",
        "Suspenseful": "fear_x",
        "Sad": "sadness_x"
    }

    if tone in tone_map:
        book_recs.sort_values(by=tone_map[tone], ascending=False, inplace=True)
    
    return book_recs.head(final_top_k)


st.title("LLM-Based Semantic Book Recommender")

with st.container():
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        user_query = st.text_input("Describe the kind of book you're looking for:", 
                                  placeholder="e.g., A story about forgiveness")
    with col2:
        categories = ["All"] + sorted(books["simple_categories"].unique().tolist())
        category_choice = st.selectbox("Category:", categories)
    with col3:
        tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
        tone_choice = st.selectbox("Emotional Tone:", tones)

    submit_button = st.button("Find Recommendations", type="primary")

st.divider()

# --- Results Display ---
if submit_button and user_query:
    recommendations = retrieve_semantic_recommendations(user_query, category_choice, tone_choice)
    
    if recommendations.empty:
        st.warning("No books found matching those criteria.")
    else:
        # Displaying results in a grid
        cols = st.columns(4) # 4 books per row
        for idx, (_, row) in enumerate(recommendations.iterrows()):
            with cols[idx % 4]:
                st.image(row["large_thumbnail"], use_container_width=True)
                
                # Author formatting logic
                authors_split = row["authors"].split(";")
                if len(authors_split) == 2:
                    authors_str = f"{authors_split[0]} and {authors_split[1]}"
                elif len(authors_split) > 2:
                    authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
                else:
                    authors_str = row["authors"]
                
                st.markdown(f"**{row['title']}**")
                st.caption(f"by {authors_str}")
                
                # Truncated description
                desc = row["description"].split()
                st.write(" ".join(desc[:25]) + "...")
else:
    st.info("Enter a description and click 'Find Recommendations' to start exploring.")