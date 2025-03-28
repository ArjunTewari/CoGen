from newsapi import NewsApiClient
from crewai import Task, Crew, Agent, Process
from  langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
from pydantic import Field
import os

_ = load_dotenv(find_dotenv())

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

from crewai_tools import PDFSearchTool
PDFSearchTool = PDFSearchTool()

from crewai_tools import BaseTool

import requests
from crewai_tools import BaseTool
from pydantic import Field

class NewsAPITool(BaseTool):
    name: str = "News search tool"
    description: str = "Fetching latest news on the topic given"

    api_key: str = Field(..., description="API key for NewsAPI")
    base_url: str = Field(default="https://newsapi.org/v2", description="Base URL for NewsAPI")

    def _run(self, query: str) -> str:
        endpoint = f"{self.base_url}/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 5,
            "apiKey": self.api_key
        }
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

class GoogleScholarTool(BaseTool):
    name: str = "Google Scholar Tool"
    description: str = "Fetches academic articles and references from Google Scholar"

    def _run(self, input_data: str) -> str:
        query = input_data
        # TODO: Integrate with a Google Scholar scraping API or official APIs if available
        return f"Research results for '{query}' from Google Scholar"

class PerplexityTool(BaseTool):
    name: str = "Perplexity Tool"
    description: str = "Queries Perplexity for real-time search and summarization"

    token: str = Field(..., description="Perplexity API token")

    def _run(self, input_data: str) -> str:
        if not self.token:
            raise ValueError("No Perplexity token provided. Please set 'token' in the tool configuration.")

        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": input_data}
            ],
            "max_tokens": 200,
            "temperature": 0.7,
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


# User Inputs
topic = input("Enter the topic: ")
content_type = input("Enter the content type (e.g., article, newsletter, etc.): ")
word_count = int(input("Enter word count: "))
tone = input("Enter tone (e.g., formal, casual, etc.): ")

# Manager Agent
manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory=(
        "You're an experienced project manager, skilled in overseeing complex projects and "
        "guiding teams to success. Your role is to coordinate the efforts of the crew members, "
        "ensuring that each task is completed on time and to the highest standard."
    ),
    allow_delegation=True,
    llm=llm,
)

# Individual Agents
Google_Agent = Agent(
    role="You are an expert researcher.",
    goal=f"Gather relevant data from reliable sources on the topic: {topic}.",
    backstory="You have advanced web scraping capabilities and knowledge of gathering relevant information available on the web and pass it to Writer_Agent.",
    llm=llm,
    tools=[GoogleScholarTool()],
)

Perplexity_Agent = Agent(
    role="You are an AI research assistant.",
    goal=f"Find verified and accurate information on the topic: {topic} and pass it to Writer_Agent.",
    backstory="You have perplexity AI tool which helps you to research and gather accurate information about the topics provided by the user.",
    llm=llm,
    tools=[PerplexityTool(token='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')],
)

News_Agent = Agent(
    role="You are a news gathering agent.",
    goal=f"Search top 5 latest and accurate news about the topic: {topic} and pass it to Writer_Agent.",
    backstory="You have NewsAPI tool to search and gather relevant and latest news about the topic.",
    tools=[NewsAPITool(api_key="00b977b511e8492ba9c69f7bbc319fc2")],
)

Writer_Agent = Agent(
    role=(
        f"Generate a {content_type} of approximately {word_count} words, with a {tone} tone, "
        f"using all the information gathered by research agents. The content should be engaging."
    ),
    goal="Return the content and pass it to the next agent according to user inputs and demands. Strictly adhere to the word count.",
    backstory="You are an expert in writing articles, newsletters, SEO blogs, and market reports.",
    llm=llm,
    tool=[],
)

Editor_Agent = Agent(
    role="You are an expert editor.",
    goal="Modify the content generated by Writer_Agent according to the inputs provided by the user.",
    backstory="You are an expert in natural language processing and editing.",
    llm=llm,
    tools=[],
)

# ------------------------------------------------------------------------
# Create separate tasks for each research agent (all asynchronous)
# ------------------------------------------------------------------------

google_research_task = Task(
    description=(
        "Google Scholar Research Task:\n"
        f"1. Collect academic and technical insights about '{topic}'.\n"
        "2. Combine all the information for use by Writer_Agent.\n"
    ),
    expected_output=(
        "A comprehensive research document containing academic insights and keyword trends for {topic}."
    ),
    agent=Google_Agent,
 # Allow parallel execution
)

perplexity_research_task = Task(
    description=(
        "Perplexity Research Task:\n"
        f"1. Gather verified and accurate information about '{topic}' using Perplexity.\n"
        "2. Provide a concise summary for Writer_Agent.\n"
    ),
    expected_output="A set of key insights, facts, and summaries from Perplexity on {topic}.",
    agent=Perplexity_Agent,
  # Allow parallel execution
)

news_research_task = Task(
    description=(
        "News Research Task:\n"
        f"1. Fetch the top 5 latest news articles about '{topic}' using NewsAPI.\n"
        "2. Summarize key points for Writer_Agent.\n"
    ),
    expected_output="A list of recent, relevant news articles and summaries for {topic}.",
    agent=News_Agent,
  # Allow parallel execution
)

# ------------------------------------------------------------------------
# Writing and Editing Tasks (sequential or parallel as you prefer)
# ------------------------------------------------------------------------
writing_task = Task(
    description=(
        "Generating Task:\n"
        f"1. Generate a detailed draft of {content_type} on '{topic}' "
        f"within {word_count} words and using a {tone} tone.\n"
        "2. Ensure the draft is clear, informative, and captivating.\n"
    ),
    expected_output=(
        f"A well-structured draft of {content_type} on {topic} that is engaging, "
        f"informative, covers multiple aspects, and adheres to the word count/tone."
    ),
    agent=Writer_Agent,
    async_execution=False,  # Typically, you'd want the writing to happen after research
)

editor_task = Task(
    description=(
        "Editing Task:\n"
        "1. Review the draft for grammatical correctness and clarity.\n"
        "2. Edit the content as per any additional human feedback.\n"
        "3. Improve readability.\n"
        "4. Return the edited version with implemented suggestions.\n"
    ),
    expected_output=(
        "An edited version of the draft with improved grammar, clarity, and readability, "
        "along with applied feedback."
    ),
    agent=Editor_Agent,
    verbose=True,
    human_input=True,
    async_execution=False  # Typically sequential after writing
)


# ------------------------------------------------------------------------
# Assemble the crew
# ------------------------------------------------------------------------
my_crew = Crew(
    agents=[Google_Agent, Perplexity_Agent, News_Agent, Writer_Agent, Editor_Agent],
    tasks=[google_research_task, perplexity_research_task, news_research_task, writing_task, editor_task],
    verbose=2,
    timeout=120,
    memory=True,
    planning=True,  # Or Process.parallel, depending on your setup
)

# Kick off the workflow
result = my_crew.kickoff()
print(result)
