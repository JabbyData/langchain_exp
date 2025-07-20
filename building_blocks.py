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
        model_id = "gpt2"
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

        prompt_text = "What is the solution of x*3=-2 ?"

        response = llm.invoke(prompt_text)
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

        # # MIstral-7B
        # model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        # model = AutoModelForCausalLM.from_pretrained(model_id)
        # pipe = pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=tokenizer,
        #     max_new_tokens=20,
        #     temperature=0.7,
        # )
        # llm = HuggingFacePipeline(pipeline=pipe)

        # chat = ChatHuggingFace(llm=llm, verbose=True)
        # messages = [
        #     SystemMessage("You are a weather forecast presentator"),
        #     HumanMessage("Tell me the weather forecast on 01/01/2015"),
        # ]
        # response = chat.invoke(messages)
        # print(response)

        # GPT2
        model_id = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        chat = ChatHuggingFace(llm=llm, verbose=True)
        messages = [
            SystemMessage("You are a weather forecast presentator"),
            HumanMessage("Tell me the weather forecast on 01/01/2015"),
        ]
        response = chat.invoke(messages)
        print(response)

    except Exception as e:
        print(e)


def chain_exp():
    try:
        login(os.environ["HUGGINGFACEHUB_API_TOKEN"])
        model_id = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        prompt = PromptTemplate.from_template("Tell me a joke about {topic}")

        out_parser = StrOutputParser()

        format_func = lambda x: x.lower()

        chain = prompt | format_func | llm | out_parser

        response = chain.invoke({"topic": "programming"})

        print(response)

    except Exception as e:
        print(e)


def complex_chain_exp():
    try:
        login(os.environ["HUGGINGFACEHUB_API_TOKEN"])
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            temperature=0.7,
        )
        llm1 = HuggingFacePipeline(pipeline=pipe)

        model_id = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100
        )
        llm2 = HuggingFacePipeline(pipeline=pipe)
        topic_prompt = PromptTemplate.from_template("Describe me the {topic}")
        topic_chain = topic_prompt | llm1 | StrOutputParser()

        analysis_prompt = PromptTemplate.from_template(
            "Analysis the following information's mood: {info}"
        )
        analysis_chain = analysis_prompt | llm2 | StrOutputParser()

        complex_chain = topic_chain | analysis_chain
        complex_analysis = complex_chain.invoke({"topic": "bitcoin volatility"})

        print("Analysis output : \n", complex_analysis)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    api_keys_path = "config.json"
    set_api_keys(api_keys_path=api_keys_path)
    # get_models()
    # chat_with_models()
    complex_chain_exp()
