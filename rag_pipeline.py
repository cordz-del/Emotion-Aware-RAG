# rag_pipeline.py
import os
from langchain import OpenAI, PromptTemplate, LLMChain
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

def build_retrieval_index(data_dir: str):
    """
    Builds a vector index from documents located in the provided directory.
    
    Args:
        data_dir (str): Directory containing text documents.
    
    Returns:
        GPTSimpleVectorIndex: The retrieval index built from the documents.
    """
    # Load documents from the specified directory.
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = GPTSimpleVectorIndex(documents)
    return index

def generate_response(user_input: str, supportive: bool, index, llm_api_key: str):
    """
    Generates an AI response based on user input and retrieval context.
    
    Args:
        user_input (str): The original query from the user.
        supportive (bool): Flag indicating if supportive modifications are needed.
        index: The retrieval index to query for context.
        llm_api_key (str): OpenAI API key.
    
    Returns:
        str: The generated response.
    """
    # Retrieve top-k similar documents.
    retrieved_docs = index.query(user_input, similarity_top_k=3)
    context = "\n".join([doc.text for doc in retrieved_docs])
    
    # Modify prompt with supportive instructions if negative emotion is detected.
    supportive_text = "Please provide a kind and supportive response." if supportive else ""
    
    # Create the final prompt.
    prompt = f"""
    Context:
    {context}
    
    User Query:
    {user_input}
    
    {supportive_text}
    
    Response:
    """
    
    # Set up the LLM chain.
    llm = OpenAI(openai_api_key=llm_api_key)
    prompt_template = PromptTemplate(template=prompt, input_variables=[])
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    response = chain.run({})
    return response
