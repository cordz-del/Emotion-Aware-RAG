# Emotion-Aware RAG

## Overview

**Emotion-Aware RAG** is a demo repository that showcases a retrieval-augmented generation (RAG) system with an emotional intelligence twist. The project integrates a Hugging Face emotion classifier, LangChain, LlamaIndex, OpenAI’s LLM, and ChromaDB to deliver responses that are both contextually relevant and emotionally supportive.

When a user inputs text, the system detects the underlying emotion. If a negative sentiment (such as sadness, fear, or anger) is detected, the system modifies the prompt to include supportive language, ensuring the AI’s response is both helpful and empathetic.

## Features

- **Emotion Detection:** Uses a pre-trained Hugging Face emotion classifier to analyze user input.
- **Supportive Response:** Modifies LLM prompts with supportive instructions when negative emotions are detected.
- **Contextual Retrieval:** Leverages a retrieval index built from local documents (using LangChain, LlamaIndex, and ChromaDB) to provide relevant context.
- **Data-Driven Insights:** Supports analytics to track user emotion trends and evaluate pipeline performance.

## Repository Structure

