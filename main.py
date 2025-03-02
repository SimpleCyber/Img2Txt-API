from flask import Flask, request, jsonify
import cv2
import easyocr
import numpy as np

app = Flask(__name__)

# Define local model path
local_model_path = "./model"

# Initialize EasyOCR Reader using local model files
reader = easyocr.Reader(
    ['en'],  # English language
    gpu=False,  # Use CPU (set True if using GPU)
    model_storage_directory=local_model_path  # Prevents downloading, uses local models
)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Hello", "SimpleCyber": "Img2Txt-API"})

@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    # Extract text
    text_results = reader.readtext(img)
    extracted_text = [text for _, text, _ in text_results]
    
    return jsonify({"extracted_text": extracted_text})

if __name__ == '__main__':
    app.run(debug=True)
