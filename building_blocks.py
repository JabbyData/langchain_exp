"""
Module to observe LangChain building blocks
"""

# System & File
import os
import json

# Workflow
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from langchain_community.llms import FakeListLLM


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


def get_models():
    try:
        # Access a true model : Mistral-7B
        login(os.environ["HUGGINGFACEHUB_API_TOKEN"])
        # model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        model_id = "GPT2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        llm = HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 100,
            },
        )

        question_with_context_template = PromptTemplate(
            "Context Information:{context}\n\n"
            "Answer this question consisely:{question}"
        )
        prompt_text = question_with_context_template.format(
            context="You are a weather forecast presentator",
            question="What is the solution of this equation x**3 = -3 ?",
        )

        lowering_func = lambda x: x.lower()
        chain_with_transfo = prompt_text | llm | lowering_func | StrOutputParser()

        response = chain_with_transfo.invoke(prompt_text)
        print(response)

        # Access a false model
        fake_llm = FakeListLLM(responses=["Hello"])
        response = fake_llm.invoke(prompt_text)
        print(response)

    except Exception as e:
        print(e)


def chat_with_models():
    try:
        login(os.environ["HUGGINGFACEHUB_API_TOKEN"])

        # MIstral-7B
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=20,
            temperature=0.7,
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        chat = ChatHuggingFace(llm=llm, verbose=True)
        messages = [
            SystemMessage("You are a weather forecast presentator"),
            HumanMessage("Tell me the weather forecast on 01/01/2015"),
        ]
        response = chat.invoke(messages)
        print(response)

        # GPT2
        # model_id = "gpt2"
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        # model = AutoModelForCausalLM.from_pretrained(model_id)
        # pipe = pipeline(
        #     "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
        # )
        # llm = HuggingFacePipeline(pipeline=pipe)

        # chat = ChatHuggingFace(llm=llm, verbose=True)
        # messages = [
        #     SystemMessage("You are a weather forecast presentator"),
        #     HumanMessage("Tell me the weather forecast on 01/01/2015"),
        # ]
        # response = chat.invoke(messages)
        # print(response)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    api_keys_path = "langchain/config.json"
    set_api_keys(api_keys_path=api_keys_path)
    # get_models()
    chat_with_models()
