# main.py
import os
from dotenv import load_dotenv
from emotion_classifier import detect_emotion
from rag_pipeline import build_retrieval_index, generate_response

def main():
    # Load environment variables (e.g., OPENAI_API_KEY)
    load_dotenv()
    llm_api_key = os.getenv("OPENAI_API_KEY")
    if not llm_api_key:
        raise ValueError("OpenAI API key not found. Please set it in the .env file.")
    
    # Get user input.
    user_input = input("Enter your query: ")
    
    # Detect emotion from the input text.
    emotion = detect_emotion(user_input)
    print(f"Detected emotion: {emotion['label']} (score: {emotion['score']:.2f})")
    
    # Determine if we should modify the prompt for supportive responses.
    supportive = emotion['label'] in ['sad', 'fear', 'anger']
    
    # Build the retrieval index (assumes a 'data' directory with documents exists).
    index = build_retrieval_index("data")
    
    # Generate and print the AI response.
    response = generate_response(user_input, supportive, index, llm_api_key)
    print("\nAI Response:")
    print(response)

if __name__ == "__main__":
    main()
