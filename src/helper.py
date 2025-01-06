from langdetect import detect
from googletrans import Translator

def detect_language(text):
    return detect(text)

def translate_text(text, target_lang="en"):
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    return translation.text
