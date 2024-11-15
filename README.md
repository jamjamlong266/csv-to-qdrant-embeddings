# CSV Text Embedding Processor

A web-based tool for processing CSV files, creating text embeddings, and storing them in Qdrant vector database.

## Features

- CSV file upload with size validation
- Column selection for text processing
- Configurable batch size for processing
- Real-time processing status and logs
- Preview processed text before download
- Vector storage in Qdrant database
- Download processed results

## Prerequisites

- Python 3.11 or higher
- Qdrant server running locally (default: localhost:6333)

## Installation

1. Clone the repository: 
bash
git clone <repository-url>
cd <repository-name>

2. Create and activate a virtual environment:
bash
On Windows
python -m venv venv
venv\Scripts\activate
On macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install required packages:
pip install -r requirements.txt
```

Contents of requirements.txt:
```
flask
pandas
numpy
transformers
torch
qdrant-client
werkzeug
```

4. Install and start Qdrant server:
- Follow instructions at [Qdrant Installation Guide](https://qdrant.tech/documentation/quick-start/)
- Make sure Qdrant is running on localhost:6333

## Project Structure

```
.
├── app.py                  # Flask web application
├── text_embeddings_web.py  # Core processing logic
├── templates/
│   └── index.html         # Web interface
├── uploads/               # Uploaded CSV files
└── outputs/               # Processed output files
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Using the Web Interface:
   - Upload a CSV file (max 16MB)
   - Enter a collection name for Qdrant storage
   - Set batch size (1-100)
   - Select columns to process
   - Click "Process and Upload to Qdrant"
   - Monitor progress in the logs
   - Preview and download processed text file

## Processing Flow

1. CSV Upload
   - File size validation
   - Header extraction
   - Row count display

2. Text Processing
   - Column selection
   - Text concatenation
   - Batch processing

3. Embedding Generation
   - Using sentence-transformers model
   - Batched embedding creation
   - Vector size: 384 dimensions

4. Storage
   - Vectors stored in Qdrant
   - Text backup in outputs/processed_texts.txt
   - Embeddings backup in outputs/embeddings.npy

## Troubleshooting

1. If Qdrant connection fails:
   - Verify Qdrant is running: `curl http://localhost:6333/health`
   - Check Qdrant logs for errors

2. If processing fails:
   - Check the application logs
   - Verify CSV file format
   - Ensure sufficient memory for batch size

3. If file upload fails:
   - Check file size (max 16MB)
   - Verify CSV format
   - Ensure uploads directory is writable

## Notes

- The application uses the 'sentence-transformers/all-MiniLM-L6-v2' model
- Embeddings are 384-dimensional vectors
- Files are processed in configurable batches
- Processed files are saved in the outputs directory
- Upload limit is set to 16MB (configurable in app.py)

## Security Considerations

- The application is configured for local use
- Implement additional security measures for production use
- Add authentication for sensitive data
- Configure CORS policies as needed

## License

jaminpie
