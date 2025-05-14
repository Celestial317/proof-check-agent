from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import getpass
from router_agent import router_agent_run
from math_agent import math_agent_run
from english_agent import english_agent_run
from factual_agent import factual_agent_run
from coding_agent import coding_agent_run
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


load_dotenv(dotenv_path=r"C:\codesVSCApril\AIMS\Proof_Check_Agent\.env")
google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key, temperature=1.0)

def main_domain(user_input):
    domain_to_use = router_agent_run(f"Find domain of {user_input}")
    
    if domain_to_use == "math":
        return math_agent_run(user_input)

    elif domain_to_use == "english":
        return english_agent_run(user_input)
    
    elif domain_to_use == "factual":
        return factual_agent_run(user_input)

    elif domain_to_use == "coding":
        return coding_agent_run(user_input)
    
    else:
        return "Not specified domain"

if __name__ == "__main__":
    while True:
        user_input = input("Enter your prompt ")
        if user_input.lower() == "exit":
            break

        output_format_prompt = PromptTemplate.from_template("""
        You are a proof-checking assistant. The following is the result of a domain check agent to help you:
        {result}

        and following is the user input:
        {user_input}
        Please summarize it in a professional and concise manner. Tell if the statement is true or false, give explanation only if asked. For false statement do mention why false within 2 sentances. If it is said that domain is not specified, then mention you dont have enough information to check the statement.
        """)


        format_chain = LLMChain(
            llm=llm,
            prompt=output_format_prompt,
        )
        domain_response = main_domain(user_input)
        formatted_result = format_chain.run({"result": domain_response, "user_input": user_input})
  
        print(f"Response: {formatted_result}")

