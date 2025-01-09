import subprocess

subprocess.check_call(["pip", "install", "torchaudio"])

import os
from PIL import Image
import torchaudio
from transformers import pipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from langdetect import detect
from googletrans import Translator

# Load API Key
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Memory Setup
memory = ConversationBufferMemory()

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Models
text_model = pipeline("text-generation", model=model_name, use_auth_token=HUGGINGFACE_API_TOKEN)
image_captioning = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", use_auth_token=HUGGINGFACE_API_TOKEN)
speech_recognition = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h", use_auth_token=HUGGINGFACE_API_TOKEN)

def detect_and_translate(text, target_lang="en"):
    lang = detect(text)
    if lang != target_lang:
        translator = Translator()  # Create an instance of Translator
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    else:
        return text  # Return the text as it is if it's already in the target language

# Unified Pipeline
def multimodal_pipeline(input_data, input_type):
    if input_type == "text":
        input_data=detect_and_translate(input_data)
        response = text_model(input_data, max_length=50)
        return response[0]["generated_text"]
    elif input_type == "image":
        image = Image.open(input_data)
        response = image_captioning(image)
        return response[0]["generated_text"]
    elif input_type == "audio":
        waveform, _ = torchaudio.load(input_data)
        response = speech_recognition(waveform)
        return response["text"]
    elif input_type == "video":
        # Placeholder for video processing logic
        return "Video processing is not yet implemented."
    else:
        return "Unsupported input type."

# Unified Context-Aware Response
def contextual_response(input_data, input_type):
    context = multimodal_pipeline(input_data, input_type)
    chain = ConversationChain(llm=text_model, memory=memory)
    response = chain.run(input=context)
    return response
