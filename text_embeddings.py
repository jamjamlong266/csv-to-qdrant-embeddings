import pandas as pd
from llama_cpp import Llama
import numpy as np
from pathlib import Path
import requests
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import UpdateStatus

def download_model(url, model_path):
    """Download the model if it doesn't exist"""
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")

def create_text_for_embedding(row):
    """Create a single text string from relevant fields"""
    text_parts = [
        f"Title: {row['title']}",
        f"Subreddit: {row['subreddit']}"
    ]
    
    if pd.notna(row['body']) and row['body']:
        text_parts.append(f"Body: {row['body']}")
    
    return " | ".join(text_parts)

def create_qdrant_collection(client, collection_name, vector_size):
    """Create a new Qdrant collection if it doesn't exist"""
    try:
        client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists")
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Created new collection: {collection_name}")

def upload_to_qdrant(client, collection_name, embeddings, texts, metadata_df):
    """Upload vectors and metadata to Qdrant"""
    points = []
    batch_size = 100
    
    for idx, (embedding, text) in enumerate(zip(embeddings, texts)):
        # Create metadata dictionary
        metadata = {
            'text': text,
            'id': str(metadata_df.iloc[idx]['id']),
            'permalink': metadata_df.iloc[idx]['permalink'],
            'date': str(metadata_df.iloc[idx]['date']),
            'subreddit': metadata_df.iloc[idx]['subreddit'],
            'title': metadata_df.iloc[idx]['title']
        }
        
        # Create point
        point = PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload=metadata
        )
        points.append(point)
        
        # Upload in batches
        if len(points) >= batch_size or idx == len(embeddings) - 1:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Uploaded batch of {len(points)} vectors")
            points = []

def main():
    # Model settings
    model_url = "https://huggingface.co/gaianet/Nomic-embed-text-v1.5-Embedding-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf"
    model_path = "models/nomic-embed-text-v1.5.f16.gguf"
    
    # Download model
    download_model(model_url, model_path)
    
    # Initialize the model
    llm = Llama(
        model_path=model_path,
        embedding=True,
        n_ctx=2048
    )
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient("localhost", port=6333)
    collection_name = "reddit_posts"
    
    # Read CSV file
    df = pd.read_csv('posts.csv')
    
    # Create embeddings
    embeddings = []
    texts = []
    
    print("Creating embeddings...")
    for idx, row in df.iterrows():
        text = create_text_for_embedding(row)
        texts.append(text)
        embedding = llm.embed(text)
        embeddings.append(embedding)
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} rows...")
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Get vector size from first embedding
    vector_size = len(embeddings[0])
    
    # Create Qdrant collection
    create_qdrant_collection(qdrant_client, collection_name, vector_size)
    
    # Upload vectors to Qdrant
    print("Uploading vectors to Qdrant...")
    upload_to_qdrant(qdrant_client, collection_name, embeddings_array, texts, df)
    
    # Save embeddings and texts locally as backup
    np.save('embeddings.npy', embeddings_array)
    with open('processed_texts_1.txt', 'w', encoding='utf-8') as f:
        for idx, text in enumerate(texts):
            f.write(f"Index {idx}: {text}\n")
    
    print(f"Processing complete! Created embeddings with shape: {embeddings_array.shape}")
    print(f"Vectors uploaded to Qdrant collection: {collection_name}")
    print("Backup files saved: 'embeddings.npy' and 'processed_texts_1.txt'")

if __name__ == "__main__":
    main() 





# See the vector collection in Qdrant