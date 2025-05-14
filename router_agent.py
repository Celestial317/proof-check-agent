from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
import os
import getpass
from dotenv import load_dotenv


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key, temperature=1.0)


search = DuckDuckGoSearchRun()
def safe_run_search(query):
    try:
        return search.run(query)
    except Exception as e:
        print(f"Error during search: {e}")
        return None

duckduck_tool = Tool.from_function(
  name="duckduckgo_search",
  func=safe_run_search,
  description='''Search Duck Duck GO for information about the statement given and find from which domain it belongs to. Respond with only one of the following: "math", "english", "factual", "coding".
if the domain is not specified, then mention "NONE"''',
)

router_prompt = PromptTemplate.from_template("""
You are a routing agent. Given the user's question, decide which domain it belongs to: math, english, factual, or coding.
Respond with only one of the following: "math", "english", "factual", "coding".
if the domain is not specified, then mention "NONE"
Only 1 word is expected in the response.
The question is:
{question}
""")
router_chain = LLMChain(
    llm=llm,
    prompt=router_prompt,
)
router_tool = Tool.from_function(
  name="Router Tool", 
  func=router_chain.run, 
  description="Useful for when you need to answer questions about the domain of the question."
)

agent = initialize_agent(
    tools=[duckduck_tool, router_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

def router_agent_run(prompt):
  result = agent.invoke({'input':prompt})
  domain = result.get('output')
  return domain
