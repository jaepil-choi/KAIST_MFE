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

# %% colab={"base_uri": "https://localhost:8080/"} id="lfUtAoQ0n66T" outputId="5e092793-c980-4f58-e7d0-d9b09e27ad11"
pip install python-dotenv

# %% id="YpbF9bbNeJu_"
# import os
# from dotenv import load_dotenv, find_dotenv
# _= load_dotenv(find_dotenv())

# ap =os.getenv("MY_VAR")


# %% id="4l9Clt5rqb7A"
# # !pip install langchain
# # !pip install openai

# %% colab={"base_uri": "https://localhost:8080/"} id="s-ImCydtuahh" outputId="c4806843-df10-4624-b15c-d2655cb3da34"
# !pip install -U langchain-openai
# !pip install langchain


# %% id="E68TvgAvqi9c"
# prompt: 어떻게 api_key를 넣는가?
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

# Load the environment variables from the .env file
load_dotenv(find_dotenv())

# Get the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# %%
api_key = 'YOUR API KEY'

# %% id="fqnH_TF0rF3d"
chat=ChatOpenAI(temperature=0,
                openai_api_key=api_key,
                )

# %% [markdown] id="O5da_Wwf7ZVL"
# Custom Tools

# %% colab={"base_uri": "https://localhost:8080/"} id="PXQaXdH2uDrb" outputId="2be44e7e-de17-4687-b98a-8bd9215d4df5"
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

template = """Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

llm_chain = LLMChain(
    prompt = prompt,
    llm = chat
)

print(llm_chain.invoke("What are Tesla's revenue in 2022?"))

# %%
# !pip install langchain-community # Install the langchain_community package

# %% colab={"base_uri": "https://localhost:8080/"} id="yaHvvdUt5q2u" outputId="12a92cd5-0d21-446f-edf5-31bc977e67fd"
# !pip install duckduckgo-search

# %% colab={"base_uri": "https://localhost:8080/", "height": 105} id="DagiNO4l5qyt" outputId="b805bfe7-094d-409d-823b-ece1d5524392"
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
search.invoke("Tesla stock price?")

# %% id="SqEx-nYc5qvE"
from langchain.tools import Tool

duckduckgo_tool = Tool(
    name = 'DuckDuckGo Search',
    func = search.run,
    description = "Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
)

# %% colab={"base_uri": "https://localhost:8080/"} id="l4NFw9DU5qrJ" outputId="5aedfb97-2d87-47ae-e28d-dd8b9e1e3045"
from langchain.agents import AgentType
from langchain.agents import load_tools, initialize_agent

tools = [duckduckgo_tool]

agent = initialize_agent(
    tools,
    chat,
    agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors = True,
    verbose = True
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 333} id="IaczkNHy5qj6" outputId="458dbdba-515b-44f5-acf0-a6ee2c70a6e4"
agent.run("What are Tesla's revenue in 2022?")

# %% [markdown] id="1vxjbDMp-jE_"
# ChatGPT as a Financial Information Extractor

# %% id="Xe7xmrIE-iSs"
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

extraction_template = """Your task:\n\
Find the value of revenue in the given content.\n\
If you can't find the value, please output "None".\n\

Example 1:\n\
The amount of Apple's annual revenue in 2021 was $365.817B.
Result: 365.817

Given content: {text}
Result:
"""
extraction_prompt = ChatPromptTemplate.from_template(extraction_template)

extraction_chain = LLMChain(
    prompt = extraction_prompt,
    llm = chat
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 784} id="LvUXJJJG-iMe" outputId="2a91bdaf-1f5d-49d5-a0fc-a9ddb09fb911"
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(chains = [agent, extraction_chain],
                                      verbose = True)

overall_chain.run("What are Tesla's revenue in 2022?")

# %% colab={"base_uri": "https://localhost:8080/", "height": 437} id="gumNagkn-iIR" outputId="26fcbc12-2737-408f-ece7-886edcf2da78"
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(chains = [agent, extraction_chain],
                                      verbose = True)

overall_chain.run("테슬러의 2022년 매출은 얼마인가?")

# %% [markdown] id="tnfaSnNu_igS"
# # 연습문제

# %% id="SKp8Nbqc-iEF"
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

template = """Your task:\n\
Find the value of Scope 1 emissions in the given content.\n\
If you can't find the value, please output "None".\n\

Example 1:\n\
TotalEnergies' latest Scope 1 emissions were 32 million metric tons\
 of carbon dioxide in 2021.
Result: 32.0

Given content: {text}
Result:
"""
prompt = ChatPromptTemplate.from_template(template)

llm_chain = LLMChain(
    prompt = prompt,
    llm = chat
)

# %% id="0Y2qnO2l-iAi"
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(
    chains = [agent, llm_chain],
    verbose = True
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["ee858599f9d541fa927b5ec084f75da4", "bf1e960a079c47b78cba1e9b2410e2b2", "8b4dd772dcf647e98b6734f7a987f1fc", "b9a0d275e2a94682b7bdd8d7e8ee1f80", "0385680c0de84c258d57eff90a36cac5", "8f09e23ae2b240ebbacf1b20b0e55c5d", "084405c29b13414ab818a6213f0dbfc3", "22500a4e17574887bb5af6e8b29755b1", "b1974d4805344feb86166672c6859b84", "a80c672023ff4a49a1a73db523a606a4", "e94a839d162846cc93b76d76cb79dbb7"]} id="tX9inYGH-h9I" outputId="95327014-c1de-4ef0-b5a4-da2c7cd668d8"
from tqdm.notebook import tqdm

list_companies = ['AT&T',
 'Apple Inc.',
 'Bank of America',
 'Boeing',
 'CVS Health',
 'Chevron Corporation',
 'Cisco',
 'Citigroup',
 'Disney',
 'Dominion Energy',
 'ExxonMobil',
 'Ford Motor Company',
 'General Electric',
 'Home Depot (The)',
 'IBM',
 'Intel',
 'JPMorgan Chase',
 'Johnson & Johnson',
 "Kellogg's",
 'McKesson',
 'Merck & Co.',
 'Microsoft',
 'Oracle Corporation',
 'Pfizer',
 'Procter & Gamble',
 'United Parcel Service',
 'UnitedHealth Group',
 'Verizon',
 'Walmart',
 'Wells Fargo']

list_results = []

for i in tqdm(range(len(list_companies))):
  try:
    response = overall_chain.run(f"What is the amount of {list_companies[i]} Scope 1 emissions?")
  except:
    response = "None"
  list_results.append(response)

# %% id="gKDSlLvZ-h5d"
list_results

# %% [markdown] id="6dIXSulFAvzn"
# Sustainability Report as a Knowledge Base

# %% colab={"base_uri": "https://localhost:8080/"} id="vfrPNNwf-h1k" outputId="728e0dfa-a4b1-4911-f36c-ce75ad2e8c61"
# !pip install unstructured
# !pip install pypdf
# !pip install pdf2image

# %% colab={"base_uri": "https://localhost:8080/", "height": 339} id="vMLYSAdOBFYp" outputId="b67cf6e7-6be2-46c7-af28-eb26ff024113"
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import OnlinePDFLoader

url = "https://www.tesla.com/ns_videos/2021-tesla-impact-report.pdf"
loader = OnlinePDFLoader(url)
data = loader.load()

# %% id="QC4epjtMBFUM"
# !pip install docarray
# !pip install tiktoken

# %% id="usrVoawfBFQq"
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

# %% id="uOCLZACPBFNT"
index.query("What is the Scope 1 Emissions?")

# %% id="Btcg91DNBFJp"

# %% id="RcxWNSDjBFFy"

# %% id="MJnxU3gpBFCK"
