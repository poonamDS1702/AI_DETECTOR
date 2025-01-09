#!/usr/bin/env python
# coding: utf-8

# In[14]:

import subprocess

subprocess.check_call(["pip", "install", "langchain", "transformers", "langdetect", "googletrans"])
subprocess.check_call(["pip", "install", "langchain_community"])

# In[65]:


from langdetect import detect
from googletrans import Translator
  
def detect_and_translate(text, target_lang="en"):
    lang = detect(text)
    if lang != target_lang:
        translator = Translator()  # Create an instance of Translator
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    else:
        return text  # Return the text as it is if it's already in the target language


# In[67]:
subprocess.check_call(["pip", "install", "python-dotenv"])


# In[92]:


import os
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub


# In[93]:


# Load environment variables from .env file
load_dotenv()
huggingface_api_token = os.getenv("HUGGINGFACE_API_KEY")


# In[112]:


## intialize the hugging face LLM 

# Initialize the Hugging Face LLM with API key
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # Replace with your preferred model
    model_kwargs={"temperature": 0.1, "max_length": 200},
    huggingfacehub_api_token=huggingface_api_token,
)


# In[113]:


def truncate_text(text, max_tokens=1400):
    # Tokenize the input and truncate to fit within the limit
    tokens = text.split()  # Basic tokenization (or use a tokenizer library for precision)
    if len(tokens) > max_tokens:
        return " ".join(tokens[:max_tokens])
    return text


# In[114]:


# Conversation memory to store context
memory = ConversationBufferMemory(memory_key="chat_history")


# In[115]:


# Rephrasing Prompt
rephrase_prompt = PromptTemplate(
    input_variables=["text"],
    template="Rephrase the following text for clarity: {text}",
)

# Final Answer Prompt
answer_prompt = PromptTemplate(
    input_variables=["text", "chat_history"],
    template=(
        "You are an intelligent assistant. Use the conversation history below:\n"
        "{chat_history}\n"
        "Now, respond to this query in detail: {text}"
    ),
)


# In[116]:


def langchain_pipeline(input_text):
    # Step 1: Detect and Translate
    translated_text = detect_and_translate(input_text)
    print("Translated Text:", translated_text)  # Debugging

    # Truncate if necessary
    truncated_text = truncate_text(translated_text)
    print("Truncated Text:", truncated_text)  # Debugging

    # Step 2: Rephrase using LangChain
    rephrase_chain = LLMChain(llm=llm, prompt=rephrase_prompt)
    rephrased_text = rephrase_chain.run(truncated_text)
    print("Rephrased Text:", rephrased_text)  # Debugging

    # Step 3: Generate Final Answer with Memory
    answer_chain = LLMChain(llm=llm, prompt=answer_prompt, memory=memory)
    final_answer = answer_chain.run({"text": rephrased_text})
    print("Final Answer:", final_answer)  # Debugging

    return final_answer