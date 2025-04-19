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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10008, "status": "ok", "timestamp": 1724940765650, "user": {"displayName": "Jae Choi", "userId": "11721422988883293277"}, "user_tz": -540} id="lfUtAoQ0n66T" outputId="212899ab-3d41-40f8-d5e1-fe13490da152"
pip install python-dotenv

# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1724940765650, "user": {"displayName": "Jae Choi", "userId": "11721422988883293277"}, "user_tz": -540} id="YpbF9bbNeJu_"
# import os
# from dotenv import load_dotenv, find_dotenv
# _= load_dotenv(find_dotenv())

# ap =os.getenv("MY_VAR")


# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1724940765650, "user": {"displayName": "Jae Choi", "userId": "11721422988883293277"}, "user_tz": -540} id="4l9Clt5rqb7A"
# # !pip install langchain
# # !pip install openai

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 22605, "status": "ok", "timestamp": 1724940788249, "user": {"displayName": "Jae Choi", "userId": "11721422988883293277"}, "user_tz": -540} id="s-ImCydtuahh" outputId="83f8d237-1a0b-43b8-a841-0c8680bf13c3"
# !pip install -U langchain-openai
# !pip install langchain


# %% executionInfo={"elapsed": 8385, "status": "ok", "timestamp": 1724940805712, "user": {"displayName": "Jae Choi", "userId": "11721422988883293277"}, "user_tz": -540} id="E68TvgAvqi9c"
# prompt: 어떻게 api_key를 넣는가?
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

# Load the environment variables from the .env file
load_dotenv(find_dotenv())

# Get the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# %%
api_key = 'YOUR_API_KEY'

# %% executionInfo={"elapsed": 1246, "status": "ok", "timestamp": 1724940931607, "user": {"displayName": "Jae Choi", "userId": "11721422988883293277"}, "user_tz": -540} id="fqnH_TF0rF3d"
url = "https://github.com/tlorans/ClimateRisks/blob/main/DEXUSEU.csv?raw=true"

import pandas as pd
data = pd.read_csv(url)
data.rename(columns = {"DEXUSEU":"Exchange Rate Dollar to Euro"}, inplace = True)
data.to_csv("exchange_rate.csv", index = None)

# %%
# !pip install langchain-community # Install the langchain_community package

# %% colab={"base_uri": "https://localhost:8080/", "height": 598} executionInfo={"elapsed": 331, "status": "error", "timestamp": 1724940934919, "user": {"displayName": "Jae Choi", "userId": "11721422988883293277"}, "user_tz": -540} id="GpX9chYVHuUI" outputId="b55017a6-f2fb-4c8d-8d0f-e29c40865f9e"
from langchain.document_loaders import CSVLoader

file = "exchange_rate.csv"
loader = CSVLoader(file_path = file)
data = loader.load()
print(data[0])

# %% [markdown] id="O5da_Wwf7ZVL"
# Custom Tools

# %% id="PXQaXdH2uDrb"
# !pip uninstall -y docarray
# !pip install docarray
# !pip install tiktoken

# %% id="yaHvvdUt5q2u"
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings # import embeddings

# Initialize embeddings object
embeddings = OpenAIEmbeddings(
    openai_api_key = api_key
)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings # add embeddings argument
).from_loaders([loader])

# %%
from langchain.llms import OpenAI # import the OpenAI class

llm = OpenAI(
    temperature=0,
    openai_api_key = api_key
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 391} id="d7M71SXrSRDe" outputId="c5345447-46a0-48bf-eaeb-05d035f9fd7a"
index.query(
    "What is the last exchange rate? Gives the date",
    llm = llm
    )

# %%
# TODO: docarray 버전 충돌 문제로 인해 실행이 안됨

# %% colab={"base_uri": "https://localhost:8080/", "height": 773} id="QnoZImdUPZIV" outputId="1a307c06-74e2-46bf-9b18-600213f0c32a"
# pip install -U docarray==0.21.0

# %% colab={"base_uri": "https://localhost:8080/", "height": 453} id="aVLDrM4eOhzD" outputId="5b1ac7ad-0488-451c-e5a7-05b3c60c2973"
from docarray import DocumentArray, Document

# query_text를 사용하여 Document 객체 생성
query_text = "What is the last exchange rate? Gives the date"
query_doc = Document(text=query_text)

# Document 객체를 DocumentArray에 추가
docs = DocumentArray([query_doc])

# DocumentArray를 사용하여 쿼리 실행
response = index.query(docs)

print(response)



# %% id="MJnxU3gpBFCK"

# %% id="K0KJyKOwOoEb"
from langchain.tools import Tool

knowledge_tool = Tool(
    name = "FX Rate",
    func = index.query,
    description = "Useful for when you need to search for the exchange rate. Provides your input as a search query."
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 651} id="PnxZJTWiOoAr" outputId="11779795-545b-4e36-fe1c-fc7e9b98cec9"
from langchain.agents import AgentType
from langchain.agents import load_tools, initialize_agent
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature = 0.)


tools = [knowledge_tool]

agent = initialize_agent(
    tools,
    chat,
    agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors = True,
    verbose = True
)

agent.run("What was the exchange rate in April 2022?")

# %% [markdown] id="YBkvOZoKVhIQ"
# Interacting with Other tools

# %% colab={"base_uri": "https://localhost:8080/"} id="5BG_65PWOnxa" outputId="fe37dcba-0161-4925-a406-978727a9a50e"
# !pip install duckduckgo-search

# %% id="VKnrtnVyOntf"
tools = load_tools(["llm-math"], llm = chat)

from langchain.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

duckduckgo_tool = Tool(
    name = 'DuckDuckGo Search',
    func = search.run,
    description = "Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
)

tools.append(knowledge_tool)
tools.append(duckduckgo_tool)

agent = initialize_agent(
    tools,
    chat,
    agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors = True,
    verbose = True
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 810} id="Ertx2us4Onpu" outputId="764ec41f-2287-45c7-df06-77915339e3bf"
query = """Process as follow:\n\
1. Search for Tesla's revenue in 2022. \n\
2. Find the exchange rate in December 2022. \n\
3. Multiply Tesla's revenue in 2022 with the exchange rate. \n\
"""
agent.run(query)

# %% id="ntEVqlVoVyps"

# %% [markdown] id="x7HXwdMTV6i-"
# 연습

# %% id="O4HaISVKVylz"
url = "	https://pasteur.epa.gov/uploads/10.23719/1528686/SupplyChainGHGEmissionFactors_v1.2_NAICS_CO2e_USD2021.csv"

import pandas as pd
emissions = pd.read_csv(url)

emissions.to_csv("emissions_factors.csv", index = None)

file = "emissions_factors.csv"
loader = CSVLoader(file_path = file)
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

# %% colab={"base_uri": "https://localhost:8080/", "height": 391} id="_Rca9wouVyh4" outputId="851a93be-1183-4b1e-9f9d-adeef2e44abd"
index.query("What is the emission factor with margins for automobile industry?")

# %% colab={"base_uri": "https://localhost:8080/"} id="i2TwSVqnVyeO" outputId="a5b360a7-01a7-4387-9676-709056bc95ea"
# !pip install openai
# !pip install langchain
# !pip install docarray
# !pip install tiktoken
# !pip install duckduckgo-search

# %% id="G4BslYIpVyaU"
from langchain.tools import Tool

knowledge_tool = Tool(
    name = "Supply Chain Emissions Factors",
    func = index.query,
    description = "Useful for when you need to find the emissions supply-chain for a specific industry. You want to return the emissions factors from the \
    most related industry."
)

# %% id="QNq9Zs9LVyWM"
from langchain.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

duckduckgo_tool = Tool(
    name = 'DuckDuckGo Search',
    func = search.run,
    description = "Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
)

# %% id="LcKQHnH8OnmF"
from langchain.agents import AgentType
from langchain.agents import load_tools, initialize_agent
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature = 0.)
tools = load_tools(["llm-math"], llm = chat)

tools.append(knowledge_tool)
tools.append(duckduckgo_tool)

agent = initialize_agent(
    tools,
    chat,
    agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors = True,
    verbose = True
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 983} id="GPikDes-WWGs" outputId="dbae0395-95db-4f5d-8562-6617c6245004"
agent.run("""Please follow the following process: \n\
1. Find Tesla's main activity and last available revenue.\n\
2. Find the emissions factor with margin corresponding to Tesla's activity\n\
3. Multiply the emissions factor with the revenue\n\
Result:
""")

# %% id="p-mWYUBXWWDA"

# %% id="2ccV1fxTWV_q"

# %% id="qDcWNZNyWV7p"

# %% id="i8yKugoRWV3U"

# %% id="8F0OLmvaWVzq"

# %% id="N9xBBpyXWVwA"
