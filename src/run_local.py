from src.helper import MultilingualHelper

def process_user_input(question, context, user_lang="auto", target_lang="en"):
    """Handle user input and provide an answer."""
    helper = MultilingualHelper()

    # Translate question and context to the model's language
    translated_question = helper.translate(question, target_lang)
    translated_context = helper.translate(context, target_lang)

    # Get the answer from the model
    answer = helper.answer_question(translated_context, translated_question)

    # Translate the answer back to the user's language
    final_answer = helper.translate(answer, user_lang)

    return final_answer
