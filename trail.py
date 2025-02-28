import cv2
import easyocr

# Read image
image_path = './image.png'
img = cv2.imread(image_path)

# Initialize EasyOCR Reader (CPU mode)
reader = easyocr.Reader(['en'], gpu=True)

# Detect text in the image
text_results = reader.readtext(img)

# Print only the extracted text
for _, text, _ in text_results:
    print(text) 
