import pytesseract
from PIL import Image
from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Convert OpenCV image to PIL format
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Extract text using Tesseract
    extracted_text = pytesseract.image_to_string(pil_img)

    return jsonify({"extracted_text": extracted_text})

if __name__ == '__main__':
    app.run(debug=True)
