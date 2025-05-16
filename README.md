---

# Multi-Agent Truth & Reasoning Checker

A modular, domain-specific **validation system** built using the **LangChain framework**. It uses external knowledge sources like **Wikipedia** and **DuckDuckGo**, along with **LLM-powered tools**, to verify the correctness of statements in various domains — including math, factual knowledge, language, and code.
This system **does not generate new answers** — it only **checks** and **validates** existing statements for accuracy and soundness.

---

## Table of Contents

* [Overview](#overview)
* [Project Structure](#project-structure)
* [Features](#features)
* [Setup Instructions](#setup-instructions)
* [Usage](#usage)
* [Sample Inputs](#sample-inputs)
* [Dependencies](#dependencies)
* [Author](#author)

---

## Overview

This system uses **LangChain**, an open framework for developing LLM-powered applications, to create a multi-agent architecture where each agent specializes in verifying different types of inputs.

The agents work collaboratively, ensuring that the main agent can intelligently delegate verification tasks, retrieve external information, and present fact-checked results with reasoning.

---

## Project Structure

```
.
├── main_agent.py        # Main controller; routes statements to specific agents
├── router_agent.py      # Identifies domain using search tools
├── math_agent.py        # Validates math-related expressions and logic
├── english_agent.py     # Checks grammar, language correctness, and clarity
├── coding_agent.py      # Audits coding claims and language usage
├── factual_agent.py     # Uses Wikipedia & DuckDuckGo for fact-checking
├── requirements.txt     # All dependencies and LLM tooling
└── README.md            # This documentation
```

---

## Features

* **LangChain-Based Agent Architecture**
  Designed using LangChain’s modular components, agents, and tool integrations.

* **Domain Routing Agent**
  Uses search-based heuristics to determine whether the claim is factual, mathematical, coding-related, or linguistic.

* **Fact-Checker**
  Queries Wikipedia and DuckDuckGo to validate real-world claims.

* **Math Reasoning Checker**
  Uses `LLMMathChain` for numerical expression verification.

* **Code Logic Validator**
  Audits basic programming knowledge across multiple languages and concepts.

* **Language Assessment Agent**
  Detects grammar errors and suggests fixes.

---

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/multi-agent-verification-framework.git
   cd multi-agent-verification-framework
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add API Keys:**
   Create a `.env` file and add:

   ```env
   GOOGLE_API_KEY=your_google_api_key
   ```

---

## Usage

Run the validation system:

```bash
python main_agent.py
```

* The system asks for user input (e.g., a claim or statement).
* It identifies the domain via `router_agent.py`.
* It then routes the input to the appropriate validation agent.
* The selected agent uses LLM + external tools to verify the truthfulness and explain the result.

---

## Sample Inputs

| Input Statement                       | Routed Agent  | Result                        |
| ------------------------------------- | ------------- | ----------------------------- |
| "Python is a compiled language."      | Factual Agent | Incorrect – it's interpreted  |
| "He go to school everyday."           | English Agent | Incorrect – grammar fixed     |
| "2 + 2 × 3 = 12"                      | Math Agent    | Incorrect – verified by order |
| "`len('hello')` returns 5 in Python." | Coding Agent  | Correct – confirmed           |

---

## Dependencies

Defined in `requirements.txt`. Includes:

* `langchain`
* `langchain-google-genai`
* `langchain-community`
* `duckduckgo-search`
* `wikipedia-api`
* `python-dotenv`

All built around the **LangChain framework**.

---

## Author

**Soumya Sourav Das**

[Portfolio](https://soumya-sourav-portfolio.vercel.app/) | [GitHub](https://github.com/Celestial317) | [LinkedIn](https://www.linkedin.com/in/soumyasouravdas/)


---
