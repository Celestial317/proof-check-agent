from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
import os
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import warnings
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key, temperature=0.0)


search = DuckDuckGoSearchRun()
def safe_run_search(query, retries=3, delay=2):
    for _ in range(retries):
        try:
            return search.run(query)
        except Exception as e:
            print(f"Error during search: {e}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    return None
duckduck_tool = Tool.from_function(
    name="DuckDuckGo",
    func=safe_run_search,
  description='''Search web for information about the mathematical problem. Give numerical answer in proper string format''',
)


math_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool.from_function(
  name="Calculator",
  func=math_chain.run,
  description='''Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.'''
  )


reasoning_prompt = PromptTemplate.from_template('''You are a reasoning agent tasked with solving the user's logic-based questions. Logically arrive at the solution, and be factual. In your answer, tell if the equation is correct or not with proof. Question  {question}. providing a clear and step-by-step explanation. Present your response in a structured point-wise format for better understanding.''')
reasoning_chain = LLMChain(
  llm=llm,
  prompt=reasoning_prompt)
reasoning_tool = Tool.from_function(
  name="Reasoning Tool", 
  func=reasoning_chain.run, 
  description="Useful for when you need to answer logic-based/reasoning questions.")



agent = initialize_agent(
    tools=[duckduck_tool, math_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

def math_agent_run(prompt):
  result = agent.invoke({'input':prompt})
  maths_answer = result.get('output')
  return maths_answer