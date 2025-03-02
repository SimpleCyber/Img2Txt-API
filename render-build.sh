#!/bin/bash

# Install system dependencies
apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev

# Install Python dependencies
pip install -r requirements.txt
