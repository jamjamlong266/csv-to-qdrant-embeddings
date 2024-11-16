import pandas as pd
import numpy as np
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoModel, AutoTokenizer
import torch
import logging
from qdrant_client.http import models
import shutil
import json
import tarfile
from datetime import datetime
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_text_for_embedding(row, selected_columns):
    """Create a single text string from selected fields"""
    text_parts = []
    
    for column in selected_columns:
        value = row[column]
        if pd.notna(value) and str(value).strip():
            text_parts.append(f"{column.capitalize()}: {value}")
    
    return " | ".join(text_parts) if text_parts else "No content available"

def create_qdrant_collection(client, collection_name, vector_size):
    """Create a new Qdrant collection if it doesn't exist"""
    try:
        client.get_collection(collection_name)
        logger.info(f"Collection {collection_name} already exists")
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info(f"Created new collection: {collection_name}")

def upload_to_qdrant(client, collection_name, embeddings, texts, metadata_df):
    """Upload vectors and metadata to Qdrant"""
    points = []
    batch_size = 100
    
    for idx, (embedding, text) in enumerate(zip(embeddings, texts)):
        metadata = {
            'text': text,
            **{col: str(metadata_df.iloc[idx][col]) for col in metadata_df.columns}
        }
        
        point = PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload=metadata
        )
        points.append(point)
        
        if len(points) >= batch_size or idx == len(embeddings) - 1:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Uploaded batch of {len(points)} vectors")
            points = []

def get_embeddings(texts, model, tokenizer, batch_size=32):
    """Get embeddings for a list of texts in batches"""
    embeddings = []
    total_texts = len(texts)
    
    for i in range(0, total_texts, batch_size):
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
            logger.info(f"Processed {min(i + batch_size, total_texts)}/{total_texts} texts...")
    
    return np.array(embeddings)

def process_csv_file(filepath, collection_name, selected_columns, batch_size=32):
    """Main processing function"""
    try:
        # Initialize model and tokenizer
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Loading model: {model_name}")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize Qdrant client with timeout settings
        logger.info("Connecting to Qdrant...")
        qdrant_client = QdrantClient(
            "localhost", 
            port=6333,
            timeout=300  # 5 minutes timeout
        )
        
        # Read CSV file
        logger.info(f"Reading CSV file: {filepath}")
        df = pd.read_csv(filepath)
        
        # Create texts for embedding
        logger.info("Creating texts for embedding...")
        texts = [create_text_for_embedding(row, selected_columns) 
                for _, row in df.iterrows()]
        
        # Get embeddings with configured batch size
        logger.info(f"Creating embeddings with batch size: {batch_size}...")
        embeddings_array = get_embeddings(texts, model, tokenizer, batch_size=batch_size)
        
        # Get vector size
        vector_size = embeddings_array.shape[1]
        logger.info(f"Vector size: {vector_size}")
        
        # Create Qdrant collection
        create_qdrant_collection(qdrant_client, collection_name, vector_size)
        
        # Upload vectors to Qdrant
        logger.info("Uploading vectors to Qdrant...")
        upload_to_qdrant(qdrant_client, collection_name, embeddings_array, texts, df)
        
        # Save embeddings and texts locally first
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Saving embeddings and texts locally...")
        np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings_array)
        with open(os.path.join(output_dir, 'processed_texts.txt'), 'w', encoding='utf-8') as f:
            for idx, text in enumerate(texts):
                f.write(f"Index {idx}: {text}\n")
        
        # Try to create snapshot
        logger.info("Creating collection snapshot...")
        snapshot_success = create_collection_snapshot(qdrant_client, collection_name)
        
        # Create compressed backup regardless of snapshot success
        logger.info("Creating compressed backup...")
        try:
            backup_path = create_compressed_backup(collection_name)
            logger.info(f"Compressed backup created at: {backup_path}")
        except Exception as backup_error:
            logger.error(f"Failed to create compressed backup: {str(backup_error)}")
        
        logger.info(f"Processing complete! Created embeddings with shape: {embeddings_array.shape}")
        logger.info(f"Vectors uploaded to Qdrant collection: {collection_name}")
        logger.info("Backup files saved in outputs directory")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in process_csv_file: {str(e)}", exc_info=True)
        raise

def create_collection_snapshot(client, collection_name, snapshot_path="snapshots", timeout=600):
    """Create a snapshot of the Qdrant collection with extended timeout"""
    try:
        os.makedirs(snapshot_path, exist_ok=True)
        logger.info(f"Created snapshot directory: {snapshot_path}")
        
        # Create snapshot using direct REST API call
        logger.info(f"Initiating snapshot creation for collection: {collection_name}")
        
        # Create snapshot
        create_url = f"http://localhost:6333/collections/{collection_name}/snapshots"
        response = requests.post(create_url, timeout=timeout)
        
        if not response.ok:
            logger.error(f"Failed to create snapshot: {response.text}")
            return False
            
        snapshot_info = response.json()
        snapshot_name = snapshot_info.get('result', {}).get('name')
        
        if not snapshot_name:
            logger.error("No snapshot name in response")
            return False
            
        logger.info(f"Snapshot created with name: {snapshot_name}")
        
        # Download the snapshot
        output_file = os.path.join(snapshot_path, f"{collection_name}_snapshot.snapshot")
        logger.info(f"Downloading snapshot to: {output_file}")
        
        download_url = f"http://localhost:6333/collections/{collection_name}/snapshots/{snapshot_name}"
        response = requests.get(download_url, stream=True, timeout=timeout)
        
        if not response.ok:
            logger.error(f"Failed to download snapshot: {response.text}")
            return False
        
        # Save the snapshot file
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        if os.path.exists(output_file):
            logger.info(f"Snapshot file successfully created at: {output_file}")
            logger.info(f"Snapshot file size: {os.path.getsize(output_file)} bytes")
        else:
            logger.error("Snapshot file was not created")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating snapshot: {str(e)}", exc_info=True)
        logger.info("Proceeding without snapshot due to error...")
        return False

def create_compressed_backup(collection_name, snapshot_path="snapshots", output_path="outputs"):
    """Create a compressed tar.gz file containing the snapshot and related data"""
    try:
        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{collection_name}_backup_{timestamp}.tar.gz"
        archive_path = os.path.join(output_path, archive_name)
        
        logger.info(f"Creating compressed backup: {archive_name}")
        
        with tarfile.open(archive_path, "w:gz") as tar:
            # Add snapshot file if it exists
            snapshot_file = os.path.join(snapshot_path, f"{collection_name}_snapshot.snapshot")
            if os.path.exists(snapshot_file):
                logger.info(f"Adding snapshot file to archive: {snapshot_file}")
                tar.add(snapshot_file, arcname=os.path.basename(snapshot_file))
            else:
                logger.warning(f"Snapshot file not found: {snapshot_file}")
            
            # Add embeddings.npy
            embeddings_file = os.path.join(output_path, "embeddings.npy")
            if os.path.exists(embeddings_file):
                logger.info(f"Adding embeddings file to archive: {embeddings_file}")
                tar.add(embeddings_file, arcname="embeddings.npy")
            else:
                logger.warning(f"Embeddings file not found: {embeddings_file}")
            
            # Add processed_texts.txt
            texts_file = os.path.join(output_path, "processed_texts.txt")
            if os.path.exists(texts_file):
                logger.info(f"Adding texts file to archive: {texts_file}")
                tar.add(texts_file, arcname="processed_texts.txt")
            else:
                logger.warning(f"Texts file not found: {texts_file}")
        
        if os.path.exists(archive_path):
            logger.info(f"Compressed backup created at: {archive_path}")
            logger.info(f"Backup file size: {os.path.getsize(archive_path)} bytes")
            return archive_path
        else:
            logger.error("Backup file was not created")
            return None
        
    except Exception as e:
        logger.error(f"Error creating compressed backup: {str(e)}", exc_info=True)
        raise

def extract_compressed_backup(backup_file, extract_path="restored_backup"):
    """Extract a compressed backup archive"""
    try:
        os.makedirs(extract_path, exist_ok=True)
        
        logger.info(f"Extracting backup from: {backup_file}")
        with tarfile.open(backup_file, "r:gz") as tar:
            tar.extractall(path=extract_path)
        
        logger.info(f"Backup extracted to: {extract_path}")
        return extract_path
        
    except Exception as e:
        logger.error(f"Error extracting backup: {str(e)}", exc_info=True)
        raise