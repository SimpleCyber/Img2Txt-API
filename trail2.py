import cv2
import easyocr

# Define image path
image_path = './image.png'
img = cv2.imread(image_path)

# Define the local model path (to prevent re-downloading)
local_model_path = "./model"

# Initialize EasyOCR Reader with local model files
reader = easyocr.Reader(
    ['en'],  # English language
    gpu=True,  # Use GPU if available
    model_storage_directory=local_model_path  # Use local model files
)

# Detect text in the image
text_results = reader.readtext(img)

# Print only the extracted text
for _, text, _ in text_results:
    print(text)
