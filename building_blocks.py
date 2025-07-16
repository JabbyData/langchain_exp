"""
Module to observe LangChain building blocks
"""

# System & File
import os
import json

# Workflow
from langchain_openai import OpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from huggingface_hub import login


def get_models():
    try:
        login(os.environ["HUGGINGFACEHUB_API_TOKEN"])
        pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
        llm = HuggingFacePipeline(pipeline=pipe)

        prompt = "What is the capital of France?"
        response = llm.invoke(prompt)
        print(response)

    except Exception as e:
        print(e)


def set_api_keys(api_keys_path: str):
    """
    Set API keys to environment variables from a JSON config file.

    Args:
        api_keys_path (str): Path to the JSON file containing API keys.

    Returns:
        None

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If 'api_keys' key is missing in the config file.
        json.JSONDecodeError: If the config file is not valid JSON.

    Example:
        >>> set_api_keys("langchain/config.json")
        API keys set !
    """
    configs = json.load(open(api_keys_path, "r"))
    for key, value in configs["api_keys"].items():
        os.environ[key] = value
    print("API keys set !")


if __name__ == "__main__":
    api_keys_path = "langchain/config.json"
    set_api_keys(api_keys_path=api_keys_path)
    get_models()
