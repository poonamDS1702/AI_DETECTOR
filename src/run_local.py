from main import langchain_pipeline

if __name__ == "__main__":
    input_text = "What is the capital of Germany?"
    result = langchain_pipeline(input_text)
    print("Response:", result)
