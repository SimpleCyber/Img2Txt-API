from flask import Flask, request, jsonify
import cv2
import easyocr
import numpy as np

app = Flask(__name__)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)

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
    
    return jsonify({'text': extracted_text})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)

