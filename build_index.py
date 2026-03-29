import pandas as pd
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_index():
    print("Loading embedding model...")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Loading dataset...")
    df = pd.read_csv("dataset.csv")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )

    documents = []
    print(f"Processing {len(df)} movies...")
    
    # Fill nan plots with empty string
    df['Plot'] = df['Plot'].fillna("")
    
    for i, row in df.iterrows():
        plot_text = str(row['Plot'])
        if not plot_text: continue
        
        chunks = text_splitter.split_text(plot_text)
        
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": row.get('Title', 'Unknown'), 
                        "year": row.get('Release Year', 0)
                    }
                )
            )
            
        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1}/{len(df)} movies")

    print(f"Created {len(documents)} document chunks. Building FAISS index...")
    
    # We build the FAISS index
    vector_store = FAISS.from_documents(documents, embeddings_model)

    print("Saving FAISS index to artifacts/movie_faiss_v3...")
    os.makedirs("artifacts", exist_ok=True)
    vector_store.save_local("artifacts/movie_faiss_v3")
    print("FAISS index successfully built and saved!")

if __name__ == "__main__":
    build_index()
