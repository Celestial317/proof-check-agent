from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
import os
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
    description='''Search the web for syntax-related information about programming languages, frameworks, or libraries. Return proper validated examples or references if available.'''
)

code_prompt = PromptTemplate.from_template('''You are a programming assistant.

Given the task: {input}

1. First, explain the logic in simple and structured bullet points.
2. Then, provide a valid and clean Python code block to solve it.
Wrap the code inside triple backticks for formatting.''')

code_chain = LLMChain(llm=llm, prompt=code_prompt)

code_tool = Tool.from_function(
    name="Code Generator",
    func=code_chain.run,
    description='''Useful for generating Python code with explanations. Input should be a well-defined coding task.'''
)

debug_prompt = PromptTemplate.from_template('''You are a debugging expert.

The user is facing an issue in their code or logic.

Debug and explain the issue clearly:
1. Identify the root cause.
2. Suggest corrections with explanation.
3. Provide corrected code if applicable.

Issue: {input}
''')

debug_chain = LLMChain(llm=llm, prompt=debug_prompt)

debug_tool = Tool.from_function(
    name="Debug Helper",
    func=debug_chain.run,
    description='''Useful for debugging coding errors or logical flaws in a code snippet. Input should include the code and the issue.'''
)

agent = initialize_agent(
    tools=[code_tool, debug_tool, duckduck_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

def coding_agent_run(prompt):
    result = agent.invoke({'input': prompt})
    coding_output = result.get('output')
    return coding_output