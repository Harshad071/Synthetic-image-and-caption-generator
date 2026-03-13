# Image Captioning AI

A Streamlit web app that generates intelligent captions for images using a CNN + Transformer model.

## Features

- Pre-loaded trained tokenizer and model files
- Upload images to generate captions
- Adjustable Top-K sampling for caption generation

## How to Run Locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   streamlit run app.py
   ```

## Deployment

This app is ready for deployment on Streamlit Cloud with pre-loaded model files.

To deploy:
1. Push this code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository and set the main file to `app.py`

## Usage

1. Upload an image
2. Adjust Top-K sampling if needed
3. Click "Generate Caption"