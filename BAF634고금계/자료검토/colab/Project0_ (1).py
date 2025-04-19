# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/"} id="lfUtAoQ0n66T" outputId="9336748b-ca09-43aa-eeaf-509915e4ea3b"
pip install python-dotenv

# %% id="YpbF9bbNeJu_"
import os
# from dotenv import load_dotenv, find_dotenv
# _= load_dotenv(find_dotenv())

# ap =os.getenv("MY_VAR")


# %% id="4l9Clt5rqb7A"
# !pip install langchain
# !pip install openai
# !pip install langchain-community # Install the langchain_community package

# %% colab={"base_uri": "https://localhost:8080/"} id="s-ImCydtuahh" outputId="06f55c70-efe9-49ba-af4c-665a02a0bb8c"
pip install -U langchain-openai

# %% id="E68TvgAvqi9c"
# prompt: 어떻게 api_key를 넣는가?

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

# Load the environment variables from the .env file
load_dotenv(find_dotenv())

# Get the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# %% colab={"base_uri": "https://localhost:8080/"} id="GBTRomSctof1" outputId="162f6713-ab25-47f1-9ab2-2fd103381244"
from langchain.llms import OpenAI 

api_key = "YOUR API KEY HERE"

# Create an OpenAI object using the API key
llm = OpenAI(temperature=0, openai_api_key=api_key)

# Use the llm object to interact with the OpenAI API
response = llm.generate(["Hello world!"])

# Print the response
print(response)

# %% id="fqnH_TF0rF3d"
from langchain.chat_models import ChatOpenAI
chat=ChatOpenAI(temperature=0,
                openai_api_key=api_key
                )

# %% id="PXQaXdH2uDrb"
from langchain.prompts import ChatPromptTemplate

template="""Question: {query} \n
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# %% id="IlHMFAn8vGLW"
question="What is 2+2?"

# %% colab={"base_uri": "https://localhost:8080/"} id="rD7fn48_vK-Z" outputId="8133f8c5-1f32-4a70-f3d6-d527480ac96a"
from langchain import LLMChain

llm_chain = LLMChain(
    prompt = prompt,
    llm=chat,
)

print(llm_chain.predict(query=question))

# %% id="teroeCufwzl_"
questions = [
    {"query":"What is 2+2?"},
    {"query":"What is 3+3?"},
    {"query":"What is 4+4?"}
]

# %% colab={"base_uri": "https://localhost:8080/"} id="ubOv9Z_rw1bh" outputId="ec9f3a6c-1360-4439-f009-74c1a5f5a858"
print(llm_chain.run(questions))

# %% [markdown] id="ihHLD0x7x2xW"
# ### Other Example: Inserting Context to ChatGPT

# %% id="ZvDZljiUw1X5"
template = """
Given the following context: {context} \n
Question: Who is Bob's wife? \n
Answer: """

prompt = ChatPromptTemplate.from_template(template)

llm_chain = LLMChain(
    prompt = prompt,
    llm = chat
)

# %% colab={"base_uri": "https://localhost:8080/"} id="998Ow0c1w1T_" outputId="3215af28-115c-45a8-f3c6-c39f198ccb31"
sentence="Bob is married to Mary"
print(llm_chain.predict(context=sentence))

# %% [markdown] id="K7CU6DT8zqRs"
# Simple Sequential Chain

# %% colab={"base_uri": "https://localhost:8080/"} id="Nxlm30tSw1QU" outputId="4cb2eaf9-62dc-4a08-b534-748ea965bfdf"
first_template = """Given the following context: {query} \n
Question: How old is John? \n
Answer:
"""

first_prompt = ChatPromptTemplate.from_template(first_template)


first_chain = LLMChain(
    prompt = first_prompt,
    llm = chat
)

first_question = "John's age is half dad's age. Dad is 42 years old."

first_response = first_chain.run(query = first_question)
print(first_response)

# %% colab={"base_uri": "https://localhost:8080/"} id="JqCNeSqCw1Mv" outputId="4339fcec-4063-46e1-bcd2-f9387e9623b6"
second_template = """Given the following context:
You are are a gift recommender. Given a person's age, \
it is your job to suggest an approapriate gift for them. \n\

Person Age: \n\
{input} \n\
Suggest gift:
"""

second_prompt = ChatPromptTemplate.from_template(second_template)

second_chain = LLMChain(
    prompt = second_prompt,
    llm = chat
)

final_response = second_chain.run(input = first_response)
print(final_response)

# %% id="QydiXsCOw1JV"
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(
    chains = [first_chain, second_chain],
    verbose = True
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 211} id="zH_3KaJYw1Fz" outputId="c32d4d55-fbac-48d9-8aad-7cb8875fa294"
overall_chain.run("John's age is half dad's age. Dad is 42 years old.")

# %% [markdown] id="rHlMPRFw0LOQ"
# Exercise

# %% colab={"base_uri": "https://localhost:8080/"} id="tGyBMup4w1Cf" outputId="2a2c3f3f-38c0-4afe-a551-e7e13bbb6344"
pip install -U langchain-openai

# %% id="9Fe3V7Zsw0-9"
# # prompt: 어떻게 api_key를 넣는가?

# from dotenv import load_dotenv, find_dotenv

# # Load the environment variables from the .env file
# load_dotenv(find_dotenv())

# # Get the OpenAI API key from the environment variables
# api_key = os.getenv("OPENAI_API_KEY")

# %% id="2T1Bn4Ev0eT1"
from langchain.chat_models import ChatOpenAI
chat=ChatOpenAI(temperature=0,
                openai_api_key=api_key
                )

# %% colab={"base_uri": "https://localhost:8080/"} id="3ZdXukOt0ePd" outputId="e75ac1ea-a465-4448-8d31-6a17836eb4c3"
pip install wikipedia

# %% colab={"base_uri": "https://localhost:8080/"} id="KbZSLutD0eLb" outputId="9ddce0b8-fa54-4d17-a98b-ba3f724a1b00"
from langchain.utilities import WikipediaAPIWrapper

wikipedia = WikipediaAPIWrapper(top_k_results = 1)
print(wikipedia.run('Tesla, Inc.'))

# %% [markdown] id="JNrTqsb806F-"
# Please generate a refined document of the following document. And please ensure that the refined document meets the following criteria:
# 1. The refined document should be abstract and does not change any original meaning of the document.
# 2. The refined document should retain all the important objects, concepts, and relationships between them.
# 3. The refined document should only contain information that is from the document.
# 4. The refined document should be readable and easy to understand without any abbreviations and misspellings.
# Here is the content: [x]

# %% [markdown] id="wdD6738J0-eN"
# You are a knowledge graph extractor, and your task is to extract and return a knowledge graph from a given text.Let’s extract it step by step:
# (1). Identify the entities in the text. An entity can be a noun or a noun phrase that refers to a real-world object or an abstract concept. You can use a named entity recognition (NER) tool or a part-of-speech (POS) tagger to identify the entities.
# (2). Identify the relationships between the entities. A relationship can be a verb or a prepositional phrase that connects two entities. You can use dependency parsing
# to identify the relationships.
# (3). Summarize each entity and relation as short as possible and remove any stop words.
# (4). Only return the knowledge graph in the triplet format: (’head entity’, ’relation’, ’tail entity’).
# (5). Most importantly, if you cannot find any knowledge, please just output: "None".
# Here is the content: [x]

# %% id="thWft8vv0eH7"
from langchain.prompts import ChatPromptTemplate


first_template = """Please generate a refined document of the following document. \n\
And please ensure that the refined document meets the following criteria: \n\
1. The refined document should be abstract and does not change any original \
meaning of the document. \n\
2. The refined document should retain all the important objects, concepts, and \
relationship between them. \n\
3. The refined document should only contain information that is from \
the document. \n\
4. The refined document should be readable and easy to understand without any \
abbrevations and misspellings. \n\
Here is the content: {content}
"""

first_prompt = ChatPromptTemplate.from_template(first_template)

# %% id="od2yrMFK1GAU"
from langchain.chains import LLMChain

first_chain = LLMChain(
    prompt = first_prompt,
    llm = chat
)

# %% id="1F17FoHT1F8c"
second_template = """You are a knowledge graph extractor, and your task is to extract\
and return a knowledge graph from a given text. Let's extract it step by step:\n\
(1). Identify the entities in the text. An entity can be a noun or noun phrase \
that refers to a real-world object or an abstract concept. You can use a named entity\
recognition (NER) tool or a part-of-speech (POS) tagger to identify the entities. \n\
(2). Identify the relationships between the entities. A relationship can be a verb \
or a prepositional phrase that connects two entities. You can use dependency parsing \
to identify the relationships. \n\
(3). Summarize each entity and relation as short as possible and remove any stop words. \n\
(4). Only return the knowledge graph in the triplet format: ('head entity', 'relation', 'tail entity'). \n\
(5). Most importantly, if you cannot find any knowledge, please just output: "None". \n\
Here is the content: {content}
"""
second_prompt = ChatPromptTemplate.from_template(second_template)

second_chain = LLMChain(
    prompt = second_prompt,
    llm = chat
)

# %% id="c1qgcyg71F40"
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(
    chains = [first_chain, second_chain],
    verbose = True
)

# %% id="QrM0GSK01F1F"
from langchain.utilities import WikipediaAPIWrapper

wikipedia = WikipediaAPIWrapper(top_k_results = 1)
wiki_pages = wikipedia.run('Tesla, Inc.')

# %% colab={"base_uri": "https://localhost:8080/", "height": 541} id="-bjk4XiX1Fxp" outputId="9ebd91fa-7e77-433c-ce74-eafaa8ba6c6a"
overall_chain.run(wiki_pages)

# %% [markdown] id="K5wsRmtA1fEf"
# Agent

# %% id="_pDaRPsQ1Ft9"
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools

chat = ChatOpenAI(temperature = 0.)

# %% id="TwQxmHzI1Fqh"
tools = load_tools(["wikipedia"], llm = chat, top_k_results = 1)

# %% colab={"base_uri": "https://localhost:8080/"} id="7F-Jav-u1mLO" outputId="c01caa29-b96e-4d62-f755-a771060ca7e9"
agent = initialize_agent(tools,
                         llm = chat,
                         agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                        handle_parsing_errors=True,
                         verbose = True
                         )

# %% colab={"base_uri": "https://localhost:8080/", "height": 506} id="L8rpRtfg1mHt" outputId="8d28e7de-6bb9-47f2-c3db-2735d356fc89"
agent.run("Tesla, Inc.")

# %% id="cIMl3Dhq5Rig"

# %% id="FXcKSEyv5Rel"
overall_chain = SimpleSequentialChain(
    chains = [agent,
              first_chain,
              second_chain],
    verbose = True
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 645} id="-_OR-dp25RbE" outputId="2af8e09c-cc84-44ae-974c-662cd92af6d5"
overall_chain.run("Tesla, Inc.")

# %% id="0fY7Wf655q96"

# %% id="0UG7WJWx5q6H"

# %% id="yaHvvdUt5q2u"

# %% id="DagiNO4l5qyt"

# %% id="SqEx-nYc5qvE"

# %% id="l4NFw9DU5qrJ"

# %% id="yBLrKEB15qnU"

# %% id="IaczkNHy5qj6"
