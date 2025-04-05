# emotion_classifier.py
from transformers import pipeline

# Initialize the emotion classifier with a pre-trained model.
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def detect_emotion(text: str) -> dict:
    """
    Detects the dominant emotion in the input text.
    
    Args:
        text (str): User input text.
    
    Returns:
        dict: The emotion label and its score.
    """
    results = emotion_classifier(text)
    # Choose the emotion with the highest score.
    dominant = max(results[0], key=lambda x: x['score'])
    return dominant

# Example usage:
if __name__ == "__main__":
    sample_text = "I feel overwhelmed and anxious today."
    result = detect_emotion(sample_text)
    print(f"Detected Emotion: {result['label']} (score: {result['score']:.2f})")
