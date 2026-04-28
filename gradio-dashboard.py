import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr
load_dotenv()
books=pd.read_csv("books_with_emotions.csv")

#fethching books thumbnails
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

#Vector Database Code
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()   #reading tagged descriptions into text loader 
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0) 
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

#function that retrieves semantic recomemendations from our books dataset and applies filtering based on category an dsorting absed on emotional tone
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    books_list = [int(rec[0].page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)
    if tone == "Happy":
     book_recs.sort_values(by="joy_x", ascending=False, inplace=True)
    elif tone == "Surprising":
     book_recs.sort_values(by="surprise_x", ascending=False, inplace=True)
    elif tone == "Angry":
     book_recs.sort_values(by="anger_x", ascending=False, inplace=True)
    elif tone == "Suspenseful":
     book_recs.sort_values(by="fear_x", ascending=False, inplace=True)
    elif tone == "Sad":
     book_recs.sort_values(by="sadness_x", ascending=False, inplace=True)
     return book_recs
    
def recomend_books(
    query:str,
    category:str,
    tone:str
):
  recommendations=retrieve_semantic_recommendations(query,category,tone)
  results=[]
  for _, row in recommendations.iterrows():
    description=row["description"]
    truncated_desc_split=description.split()
    truncated_description=" ".join(truncated_desc_split[:30])+ "..."
    authors_split=row["authors"].split(";")
    if len(authors_split)==2:
      authors_str=f"{authors_split[0]} and {authors_split[1]}"
    elif len(authors_split) > 2:
     authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
    else:
     authors_str = row["authors"]
    caption = f"{row['title']} by {authors_str}: {truncated_description}"
    results.append((row["large_thumbnail"], caption))
  return results
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

#gradio dashboard
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# LLLM  based semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recomend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)

if __name__ == "__main__":
  dashboard.launch(server_name="0.0.0.0", share=True)