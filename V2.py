
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import requests
from dotenv import find_dotenv, load_dotenv
import uuid
from langchain_community.callbacks import get_openai_callback

# Load environment variables
load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# Import crewai and langchain modules (assumed available)
from crewai import Task, Crew, Agent
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Tools
from crewai_tools import  BaseTool

# -------------------------------
# Define tools for the agents
# -------------------------------
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

# -------------------------------
# Define the request models
# -------------------------------
class GenerationRequest(BaseModel):
    topic: str
    content_type: str
    word_count: int
    tone: str
    additional_requirements: str = ""

class FinalFeedbackRequest(BaseModel):
    draft_content: str
    feedback: str

# -------------------------------
# Initialize the FastAPI application
# -------------------------------
app = FastAPI()

# Enable CORS (adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pending_crews = {}

# -------------------------------
# API Endpoint to generate content
# -------------------------------
@app.post("/generate-content")
def generate_content(request: GenerationRequest):
    try:
        # Extract request parameters
        topic = request.topic
        content_type = request.content_type
        word_count = request.word_count
        tone = request.tone
        additional_requirements = request.additional_requirements

        # -------------------------------
        # Define Agents and their roles
        # -------------------------------

        Google_Agent = Agent(
            role="You are an expert researcher.",
            goal=f"Gather relevant data from reliable sources on the topic: {topic}.",
            backstory="You have advanced web scraping capabilities and knowledge of gathering relevant information available on the web and pass it to Writer_Agent.",
            tools=[GoogleScholarTool()],
        )

        Perplexity_Agent = Agent(
            role="You are an AI research assistant.",
            goal=f"Find verified and accurate information on the topic: {topic} and pass it to Writer_Agent.",
            backstory="You have the Perplexity AI tool which helps you research and gather accurate information about the topics provided by the user.",
            tools=[PerplexityTool(token='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')],
        )

        News_Agent = Agent(
            role="You are a news gathering agent.",
            goal=f"Search top 5 latest and accurate news about the topic: {topic} and pass it to Writer_Agent.",
            backstory="You have the NewsAPI tool to search and gather relevant and the latest news about the topic.",
            tools=[NewsAPITool(api_key="00b977b511e8492ba9c69f7bbc319fc2")],
        )

        writer_role = f"Generate a {content_type} of approximately {word_count} words with a {tone} tone"
        if additional_requirements:
            writer_role += f". Additional requirements: {additional_requirements}"
        Writer_Agent = Agent(
            role=writer_role,
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

        # -------------------------------
        # Define Tasks for each Agent
        # -------------------------------
        google_research_task = Task(
            description=(
                f"Google Scholar Research Task:\n"
                f"1. Collect academic and technical insights about '{topic}'.\n"
                "2. Combine all the information for use by Writer_Agent.\n"
            ),
            expected_output=f"A comprehensive research document containing academic insights and keyword trends for {topic}.",
            agent=Google_Agent,
        )

        perplexity_research_task = Task(
            description=(
                f"Perplexity Research Task:\n"
                f"1. Gather verified and accurate information about '{topic}' using Perplexity.\n"
                "2. Provide a concise summary for Writer_Agent.\n"
            ),
            expected_output=f"A set of key insights, facts, and summaries from Perplexity on {topic}.",
            agent=Perplexity_Agent,
        )

        news_research_task = Task(
            description=(
                f"News Research Task:\n"
                f"1. Fetch the top 5 latest news articles about '{topic}' using NewsAPI.\n"
                "2. Summarize key points for Writer_Agent.\n"
            ),
            expected_output=f"A list of recent, relevant news articles and summaries for {topic}.",
            agent=News_Agent,
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
                "An edited version of the draft with improved grammar, clarity, and readability, along with applied feedback."
            ),
            agent=Editor_Agent,
            verbose=True,
            human_input=True,
            async_execution=False,
        )

        # -------------------------------
        # Assemble the crew and start the workflow
        # -------------------------------
        my_crew = Crew(
            agents=[Google_Agent, Perplexity_Agent, News_Agent, Writer_Agent, Editor_Agent],
            tasks=[google_research_task, perplexity_research_task, news_research_task, writing_task, editor_task],
            verbose=2,
            timeout=120,
            memory=True,
            planning=True,
        )

        # Generate a unique report ID and store the crew so we can resume with human feedback later
        report_id = str(uuid.uuid4())
        pending_crews[report_id] = my_crew
        with get_openai_callback() as cb:
            result = my_crew.kickoff()
            input_tokens = cb.prompt_tokens
            output_tokens = cb.completion_tokens
            input_cost = (input_tokens / 1_000_000) * 0.15
            output_cost = (output_tokens / 1_000_000) * 0.6
            cost_usd = input_cost + output_cost
            total_tokens = cb.total_tokens
        return {
            "reportId": result,
            "message": "Content generation in progress or completed.",
            "tokens_used": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "approx_cost_usd": cost_usd
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# API Endpoint to fetch final feedback and resume the human input task
# -------------------------------
@app.post("/final-feedback")
def final_feedback(request: FinalFeedbackRequest):
    try:
        crew = pending_crews.get(request.reportId)
        if crew is None:
            raise HTTPException(status_code=404, detail="Report not found or already finalized.")

        # Provide the human feedback to resume the editing task.
        # The following method is assumed to be provided by CrewAI for resuming human input tasks.
        final_content = crew.provide_human_input(request.feedback)

        # Optionally, remove the crew reference as it is no longer pending.
        del pending_crews[request.reportId]

        return {"final_content": final_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Run the application locally
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
