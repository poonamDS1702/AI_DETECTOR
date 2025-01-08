from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class MultilingualHelper:
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        self.translator = Translator()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def translate(self, text, target_lang="en"):
        """Translate text to the target language."""
        translation = self.translator.translate(text, dest=target_lang)
        return translation.text

    def answer_question(self, context, question):
        """Answer the question based on the given context."""
        inputs = self.tokenizer.encode_plus(question, context, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = self.tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
        return answer
