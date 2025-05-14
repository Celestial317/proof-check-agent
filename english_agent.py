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
    description='''Search the web for English vocabulary, spelling, grammar, idioms, and translations. Give validated information in proper string format.'''
)

grammar_prompt = PromptTemplate.from_template(
    '''You are an English Teacher. The user will provide an English sentence as input.
Output "Correct" if it is grammatically right, else output "Wrong" and rewrite the corrected sentence in the next line.
Sentence: {input}'''
)
grammar_chain = LLMChain(llm=llm, prompt=grammar_prompt)
grammar_tool = Tool.from_function(
    name="Grammar Checker",
    func=grammar_chain.run,
    description="Checks grammar correctness. Input: an English sentence."
)


vocab_prompt = PromptTemplate.from_template(
    '''Paraphrase the given sentence using more advanced or formal vocabulary without changing its meaning, or check for proper vocabulary.
Sentence: {input}'''
)
vocab_chain = LLMChain(llm=llm, prompt=vocab_prompt)
vocab_tool = Tool.from_function(
    name="Vocabulary Enhancer",
    func=vocab_chain.run,
    description="Enhances vocabulary in a sentence using formal/advanced words."
)


meaning_prompt = PromptTemplate.from_template(
    '''Explain the word "{input}" with:
1. Definition
2. Part of speech
3. Example sentence
4. Common synonyms and antonyms'''
)
meaning_chain = LLMChain(llm=llm, prompt=meaning_prompt)
meaning_tool = Tool.from_function(
    name="Word Meaning Explainer",
    func=meaning_chain.run,
    description="Provides meaning, part of speech, usage example, synonyms and antonyms for an English word."
)

idiom_prompt = PromptTemplate.from_template(
    '''Explain the idiom or phrase "{input}" with:
1. Meaning
2. Example sentence
3. Typical context where it's uses'''
)
idiom_chain = LLMChain(llm=llm, prompt=idiom_prompt)
idiom_tool = Tool.from_function(
    name="Idiom Interpreter",
    func=idiom_chain.run,
    description="Explains idioms or phrases with meaning, usage, and examples."
)


translate_prompt = PromptTemplate.from_template(
    '''Translate the following English sentence to the specified target language mentioned at the end:
Sentence: {input}
then check if the translated sentence matches the input'''
)
translate_chain = LLMChain(llm=llm, prompt=translate_prompt)
translate_tool = Tool.from_function(
    name="Translator",
    func=translate_chain.run,
    description="Translate English to any language. Format: 'Translate <sentence> to <target language>'."
)

english_agent = initialize_agent(
    tools=[duckduck_tool, grammar_tool, vocab_tool, meaning_tool, idiom_tool, translate_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

def english_agent_run(prompt):
    result = english_agent.invoke({'input': prompt})
    return result.get('output')
