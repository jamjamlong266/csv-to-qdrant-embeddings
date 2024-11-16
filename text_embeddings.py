import pandas as pd
import numpy as np
from pathlib import Path
import requests
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import UpdateStatus
from transformers import AutoModel, AutoTokenizer
import torch

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

def get_embeddings(texts, model, tokenizer, batch_size=32):
    """Get embeddings for a list of texts in batches"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize texts
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Get model output
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        # Use [CLS] token embeddings
        batch_embeddings = model_output.last_hidden_state[:, 0, :].numpy()
        embeddings.extend(batch_embeddings)
        
        if (i + batch_size) % 100 == 0:
            print(f"Processed {i + batch_size} texts...")
    
    return np.array(embeddings)

def main():
    # Initialize model and tokenizer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient("localhost", port=6333)
    collection_name = "new_reddit_posts"
    
    # Read CSV file
    df = pd.read_csv('posts.csv')
    
    # Create texts for embedding
    print("Creating texts for embedding...")
    texts = [create_text_for_embedding(row) for _, row in df.iterrows()]
    
    # Get embeddings
    print("Creating embeddings...")
    embeddings_array = get_embeddings(texts, model, tokenizer)
    
    # Get vector size
    vector_size = embeddings_array.shape[1]
    
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