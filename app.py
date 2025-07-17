import os
import requests
from transformers import MarianMTModel, MarianTokenizer, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io
import gradio as gr

# Load the MarianMT model and tokenizer for translation (Tamil to English)
model_name = "Helsinki-NLP/opus-mt-mul-en"
translation_model = MarianMTModel.from_pretrained(model_name)
translation_tokenizer = MarianTokenizer.from_pretrained(model_name)

# Load GPT-Neo for creative text generation
text_generation_model_name = "EleutherAI/gpt-neo-1.3B"
text_generation_model = AutoModelForCausalLM.from_pretrained(text_generation_model_name)
text_generation_tokenizer = AutoTokenizer.from_pretrained(text_generation_model_name)

# Add padding token to GPT-Neo tokenizer if not present
if text_generation_tokenizer.pad_token is None:
    text_generation_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the API key from the environment variable
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

# Query Hugging Face API to generate image
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

# Translate Tamil text to English
def translate_text(tamil_text):
    inputs = translation_tokenizer(tamil_text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = translation_model.generate(**inputs)
    translation = translation_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translation

# Generate an image based on the translated text
def generate_image(prompt):
    image_bytes = query({"inputs": prompt})
    image = Image.open(io.BytesIO(image_bytes))
    return image

# Generate creative text based on the translated English text
def generate_creative_text(translated_text):
    inputs = text_generation_tokenizer(translated_text, return_tensors="pt", padding=True, truncation=True)
    generated_tokens = text_generation_model.generate(**inputs, max_length=100)
    creative_text = text_generation_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return creative_text

# Function to handle the full workflow
def translate_generate_image_and_text(tamil_text):
    # Step 1: Translate Tamil to English
    translated_text = translate_text(tamil_text)
    
    # Step 2: Generate an image from the translated text
    image = generate_image(translated_text)
    
    # Step 3: Generate creative text from the translated text
    creative_text = generate_creative_text(translated_text)
    
    return translated_text, creative_text, image

# Create Gradio interface
interface = gr.Interface(
    fn=translate_generate_image_and_text,
    inputs=gr.Textbox(label="Enter Tamil Text"),  # Input for Tamil text
    outputs=[
        gr.Textbox(label="Translated Text"),        # Output for translated text
        gr.Textbox(label="Creative Generated Text"),# Output for creative text
        gr.Image(label="Generated Image")           # Output for generated image
    ],
    title="Tamil to English Translation, Image Generation & Creative Text",
    description="Enter Tamil text to translate to English, generate an image, and create creative text based on the translation."
)

# Launch Gradio app
interface.launch()
