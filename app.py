from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import os
from text_embeddings_web import process_csv_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check file size
    file_size = request.content_length
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"]/(1024*1024)}MB'}), 400
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read CSV headers and get row count
        df = pd.read_csv(filepath)
        headers = df.columns.tolist()
        row_count = len(df)
        
        return jsonify({
            'headers': headers,
            'filepath': filepath,
            'row_count': row_count,
            'file_size_mb': round(file_size/(1024*1024), 2)
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    filepath = data.get('filepath')
    collection_name = data.get('collection_name')
    selected_columns = data.get('selected_columns', [])
    
    if not all([filepath, collection_name, selected_columns]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        process_csv_file(filepath, collection_name, selected_columns)
        
        # Check if processed file exists
        processed_file = os.path.join(app.config['OUTPUT_FOLDER'], 'processed_texts.txt')
        if os.path.exists(processed_file):
            return jsonify({
                'success': True, 
                'message': 'Processing complete',
                'collection_name': collection_name,
                'has_processed_file': True
            })
        else:
            return jsonify({
                'success': True, 
                'message': 'Processing complete but no output file generated',
                'collection_name': collection_name,
                'has_processed_file': False
            })
    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}", exc_info=True)
        return jsonify({
            'error': f"Processing failed: {str(e)}"
        }), 500

@app.route('/download_processed')
def download_processed():
    """Download the processed texts file"""
    try:
        return send_file(
            os.path.join(app.config['OUTPUT_FOLDER'], 'processed_texts.txt'),
            as_attachment=True,
            download_name='processed_texts.txt'
        )
    except Exception as e:
        return jsonify({'error': 'File not found or error in download'}), 404

@app.route('/preview_processed')
def preview_processed():
    try:
        preview_lines = []
        with open(os.path.join(app.config['OUTPUT_FOLDER'], 'processed_texts.txt'), 'r', encoding='utf-8') as f:
            # Read first 10 lines for preview
            for _ in range(10):
                line = f.readline().strip()
                if not line:
                    break
                preview_lines.append(line)
        
        # Get file size
        file_size = os.path.getsize(os.path.join(app.config['OUTPUT_FOLDER'], 'processed_texts.txt'))
        file_size_mb = round(file_size / (1024 * 1024), 2)
        
        return jsonify({
            'preview': preview_lines,
            'file_size': f"{file_size_mb} MB",
            'total_lines': sum(1 for _ in open(os.path.join(app.config['OUTPUT_FOLDER'], 'processed_texts.txt')))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    app.run(debug=True) 