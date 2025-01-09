from main_1 import contextual_response

# Example Usage
if __name__ == "__main__":
    input_text = "What is the capital of Germany?"
    print(contextual_response(input_text, "text"))

    input_image_path = r"C:\Users\This PC\AI_DETECTOR\369A2959.JPG.jpg"
    print(contextual_response(input_image_path, "image"))

    #input_audio_path = "path_to_audio.wav"
    #print(contextual_response(input_audio_path, "audio"))