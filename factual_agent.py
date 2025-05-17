from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langcahin_tavily import TavilySearch
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
import os
import warnings
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key, temperature=0.0)


search = TavilySearch(max_results = 5, topic = "general")
search_tool = Tool.from_function(
    name="Tavily",
    func=search.run,
    description="Use this to get real time data and facts from Web Scraping"
)


wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
def safe_wiki(query):
    try:
        return wiki.run(query)
    except Exception as e:
        print(f"Wikipedia error: {e}")
        return None
wikipedia_tool = Tool.from_function(
    name="Wikipedia",
    func=safe_wiki,
    description="Use this to get encyclopedic or historical factual information from Wikipedia."
)

factcheck_prompt = PromptTemplate.from_template(
    '''You are a factual verifier. Cross-check the input statement using facts. Clearly say whether the statement is TRUE or FALSE.
Then give a justification using known facts.
You can use the tools DuckDuckGo and Wikipedia to find information.
Wikipedia is a good source for encyclopedic or historical information.
DuckDuckGo is a good source for real-time facts, news, statistics or events.

Statement: {input}'''
)

factcheck_chain = LLMChain(llm=llm, prompt=factcheck_prompt)
factual_tool = Tool.from_function(
    name="Fact Checker",
    func=factcheck_chain.run,
    description="Analyze and determine if a statement is true or false with reasoning."
)

factual_agent = initialize_agent(
    tools=[duckduck_tool, wikipedia_tool, factual_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

def factual_agent_run(statement):
    result = factual_agent.invoke({'input': statement})
    return result.get('output')
