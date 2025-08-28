from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import numpy as np
from solver import adjust_and_ocr_simple, adjust_and_ocr
import traceback

app = Flask(__name__)
CORS(app)

def clean_result(text):
    """
    Clean the OCR result to return only numbers without commas or spaces
    """
    if not text:
        return ""
    
    cleaned = ''.join(char for char in text if char.isdigit())
    return cleaned

@app.route('/solve', methods=['POST'])
def solve_captcha():
    """
    Main endpoint to solve captcha images
    Accepts:
    - file: image file upload
    - image_data: base64 encoded image
    - url: base64 data URL or regular image URL
    - image_url: image URL (alternative key)
    """
    try:
        result = ""
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename:
                image_data = file.read()
                result = adjust_and_ocr_simple(image_data)
        
        elif 'image_data' in request.json:
            image_data = request.json['image_data']
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            
            image_bytes = base64.b64decode(image_data)
            result = adjust_and_ocr_simple(image_bytes)
        
        elif 'url' in request.json:
            url_data = request.json['url']
            
            if url_data.startswith('data:'):
                if ',' in url_data:
                    image_data = url_data.split(',', 1)[1]
                    image_bytes = base64.b64decode(image_data)
                    result = adjust_and_ocr_simple(image_bytes)
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid data URL format'
                    }), 400
            else:
                result = adjust_and_ocr_simple(url_data)
        
        elif 'image_url' in request.json:
            image_url = request.json['image_url']
            result = adjust_and_ocr_simple(image_url)
        
        else:
            return jsonify({
                'success': False,
                'error': 'No image data provided. Send either a file upload, base64 image_data, url (including data URLs), or image_url'
            }), 400
        
        cleaned_result = clean_result(result)
        
        return jsonify({
            'success': True,
            'result': cleaned_result,
            'raw_result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'captcha-solver'
    })

@app.route('/', methods=['GET'])
def index():
    """Simple info page"""
    return jsonify({
        'service': 'Captcha Solver API',
        'endpoints': {
            'POST /solve': 'Solve captcha (accepts file upload, base64, or URL)',
            'GET /health': 'Health check'
        },
        'usage': {
            'file_upload': 'Send image file in multipart/form-data',
            'base64': 'Send {"image_data": "base64_string"} in JSON',
            'data_url': 'Send {"url": "data:image/png;base64,iVBORw0KGgo..."} in JSON',
            'url': 'Send {"url": "https://example.com/image.jpg"} in JSON',
            'image_url': 'Send {"image_url": "https://example.com/image.jpg"} in JSON'
        }
    })

if __name__ == '__main__':
    print("Starting Captcha Solver Server...")
    print("Server will be available at http://localhost:5000")
    print("Use POST /solve to solve captchas")
    app.run(host='0.0.0.0', port=5000, debug=True)
