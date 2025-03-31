import os
from dotenv import load_dotenv
import asyncio
from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import uuid
from typing import Optional, Dict, Any, List
import firebase_admin
from firebase_admin import credentials, firestore, auth
from datetime import datetime
import json
from langchain_community.callbacks import get_openai_callback

# Load environment variables from .env file
load_dotenv()

# Update the Firebase initialization to use environment variables if available
cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")
if os.path.exists(cred_path):
  cred = credentials.Certificate(cred_path)
else:
  # For deployment, we might need to use environment variables directly
  cred = credentials.Certificate({
      "type": "service_account",
      "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
      "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
      "private_key": os.environ.get("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
      "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
      "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token",
      "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
      "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_X509_CERT_URL")
  })
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# Import crewai and langchain modules
from crewai import Task, Crew, Agent
from langchain_openai import ChatOpenAI
from crewai.callbacks import CrewAgentCallbackHandler
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Tools
from crewai_tools import BaseTool

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
      # In a production environment, you would integrate with a proper Google Scholar API
      # or use a service like Serpapi
      response = requests.get(f"https://serpapi.com/search.json?engine=google_scholar&q={query}&api_key={os.environ.get('SERPAPI_API_KEY')}")
      if response.status_code == 200:
          return response.json()
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
# Custom Callback Handler for tracking agent progress
# -------------------------------
class AgentProgressCallback(CrewAgentCallbackHandler):
  def __init__(self, report_id: str):
      super().__init__()
      self.report_id = report_id
      self.agent_outputs = {}
      self.current_agent = None
      self.progress_updates = []
      
  def on_agent_start(self, agent: Agent) -> None:
      self.current_agent = agent.name or agent.role.split()[0]
      update = {
          "timestamp": datetime.now().isoformat(),
          "agent": self.current_agent,
          "status": "started",
          "message": f"{self.current_agent} has started working"
      }
      self.progress_updates.append(update)
      
      # Update Firestore with progress
      try:
          content_query = db.collection("content").where("reportId", "==", self.report_id).limit(1)
          content_docs = list(content_query.stream())
          if content_docs:
              content_ref = db.collection("content").document(content_docs[0].id)
              content_ref.update({
                  "agentProgress": firestore.ArrayUnion([update])
              })
      except Exception as e:
          print(f"Error updating progress: {e}")
  
  def on_agent_finish(self, agent: Agent, output: str) -> None:
      self.agent_outputs[self.current_agent] = output
      update = {
          "timestamp": datetime.now().isoformat(),
          "agent": self.current_agent,
          "status": "finished",
          "message": f"{self.current_agent} has completed its task"
      }
      self.progress_updates.append(update)
      
      # Update Firestore with progress and intermediate output
      try:
          content_query = db.collection("content").where("reportId", "==", self.report_id).limit(1)
          content_docs = list(content_query.stream())
          if content_docs:
              content_ref = db.collection("content").document(content_docs[0].id)
              
              # Store intermediate results for each agent
              field_name = f"intermediate_{self.current_agent.lower().replace(' ', '_')}"
              update_data = {
                  "agentProgress": firestore.ArrayUnion([update]),
                  field_name: output
              }
              
              # If this is the Writer Agent, update the draft content
              if "Writer" in self.current_agent:
                  update_data["draftContent"] = output
              
              content_ref.update(update_data)
      except Exception as e:
          print(f"Error updating progress: {e}")

# -------------------------------
# Firebase Authentication Middleware
# -------------------------------
async def verify_firebase_token(request: Request):
  auth_header = request.headers.get("Authorization")
  if not auth_header or not auth_header.startswith("Bearer "):
      raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
  
  token = auth_header.split("Bearer ")[1]
  try:
      decoded_token = auth.verify_id_token(token)
      return decoded_token
  except Exception as e:
      raise HTTPException(status_code=401, detail=f"Invalid authentication token: {str(e)}")

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
  reportId: str
  feedback: str

# -------------------------------
# Initialize the FastAPI application
# -------------------------------
app = FastAPI()

# Enable CORS
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
async def generate_content(request: GenerationRequest, user_data: dict = Depends(verify_firebase_token)):
  try:
      # Get user ID from the token
      user_id = user_data["uid"]
      
      # Check user's word allowance in Firebase
      user_ref = db.collection("users").document(user_id)
      user_doc = user_ref.get()
      
      if not user_doc.exists:
          raise HTTPException(status_code=404, detail="User not found")
      
      user_data = user_doc.to_dict()
      words_remaining = user_data.get("wordsRemaining", 0)
      free_trial_used = user_data.get("freeTrialUsed", False)
      
      # Check if this is a free trial generation
      is_free_trial = not free_trial_used and user_data.get("plan") == "free"
      
      # Check if user has enough words remaining (skip check for free trial)
      if not is_free_trial and words_remaining < request.word_count:
          raise HTTPException(status_code=403, detail="Insufficient word allowance")
      
      # Extract request parameters
      topic = request.topic
      content_type = request.content_type
      word_count = request.word_count
      tone = request.tone
      additional_requirements = request.additional_requirements

      # Generate a unique report ID
      report_id = str(uuid.uuid4())
      
      # Create a new content document in Firestore first to track progress
      content_ref = db.collection("content").document()
      content_data = {
          "userId": user_id,
          "topic": topic,
          "contentType": content_type,
          "wordCount": word_count,
          "tone": tone,
          "additionalRequirements": additional_requirements,
          "reportId": report_id,
          "status": "generating",
          "agentProgress": [],
          "createdAt": firestore.SERVER_TIMESTAMP,
      }
      content_ref.set(content_data)
      
      # Create a callback handler to track progress
      progress_callback = AgentProgressCallback(report_id)
      
      # -------------------------------
      # Define Agents and their roles
      # -------------------------------
      Google_Agent = Agent(
          name="Google Scholar Agent",
          role="You are an expert researcher.",
          goal=f"Gather relevant data from reliable sources on the topic: {topic}.",
          backstory="You have advanced web scraping capabilities and knowledge of gathering relevant information available on the web and pass it to Writer_Agent.",
          tools=[GoogleScholarTool()],
      )

      Perplexity_Agent = Agent(
          name="Perplexity Agent",
          role="You are an AI research assistant.",
          goal=f"Find verified and accurate information on the topic: {topic} and pass it to Writer_Agent.",
          backstory="You have the Perplexity AI tool which helps you research and gather accurate information about the topics provided by the user.",
          tools=[PerplexityTool(token=os.environ.get("PERPLEXITY_API_KEY"))],
      )

      News_Agent = Agent(
          name="News Agent",
          role="You are a news gathering agent.",
          goal=f"Search top 5 latest and accurate news about the topic: {topic} and pass it to Writer_Agent.",
          backstory="You have the NewsAPI tool to search and gather relevant and the latest news about the topic.",
          tools=[NewsAPITool(api_key=os.environ.get("NEWS_API_KEY"))],
      )

      writer_role = f"Generate a {content_type} of approximately {word_count} words with a {tone} tone"
      if additional_requirements:
          writer_role += f". Additional requirements: {additional_requirements}"
      Writer_Agent = Agent(
          name="Writer Agent",
          role=writer_role,
          goal="Return the content and pass it to the next agent according to user inputs and demands. Strictly adhere to the word count.",
          backstory="You are an expert in writing articles, newsletters, SEO blogs, and market reports.",
          llm=llm,
          tools=[],
      )

      Editor_Agent = Agent(
          name="Editor Agent",
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
          callbacks=[progress_callback],
      )

      # Store the crew so we can resume with human feedback later
      pending_crews[report_id] = my_crew
      
      # Run the crew in a background task to avoid blocking the API
      async def run_crew_background():
          try:
              with get_openai_callback() as cb:
                  draft_content = my_crew.kickoff()
                  input_tokens = cb.prompt_tokens
                  output_tokens = cb.completion_tokens
                  input_cost = (input_tokens / 1_000_000) * 0.15
                  output_cost = (output_tokens / 1_000_000) * 0.6
                  cost_usd = input_cost + output_cost
                  total_tokens = cb.total_tokens
              
              # Update the content document in Firestore with the draft content
              content_ref.update({
                  "draftContent": draft_content,
                  "status": "draft",
                  "tokensUsed": total_tokens,
                  "inputTokens": input_tokens,
                  "outputTokens": output_tokens,
                  "approxCostUsd": cost_usd
              })
              
              # Update user's words remaining and free trial status
              update_data = {
                  "contentGenerated": firestore.Increment(1)
              }
              
              if is_free_trial:
                  update_data["freeTrialUsed"] = True
              else:
                  update_data["wordsRemaining"] = firestore.Increment(-word_count)
              
              user_ref.update(update_data)
              
          except Exception as e:
              # Update the content document with error status
              content_ref.update({
                  "status": "error",
                  "error": str(e)
              })
              print(f"Error in background task: {e}")
      
      # Start the background task
      asyncio.create_task(run_crew_background())
      
      return {
          "reportId": report_id,
          "message": "Content generation started. You can check the progress and draft content.",
          "contentId": content_ref.id
      }
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# API Endpoint to get generation progress
# -------------------------------
@app.get("/generation-progress/{report_id}")
async def get_generation_progress(report_id: str, user_data: dict = Depends(verify_firebase_token)):
  try:
      user_id = user_data["uid"]
      
      # Get the content document from Firestore
      content_query = db.collection("content").where("reportId", "==", report_id).limit(1)
      content_docs = list(content_query.stream())
      
      if not content_docs:
          raise HTTPException(status_code=404, detail="Content not found")
      
      content_data = content_docs[0].to_dict()
      
      # Verify this content belongs to the current user
      if content_data["userId"] != user_id:
          raise HTTPException(status_code=403, detail="You don't have permission to access this content")
      
      # Return the progress data
      return {
          "status": content_data.get("status", "generating"),
          "progress": content_data.get("agentProgress", []),
          "draftContent": content_data.get("draftContent", ""),
          "intermediateResults": {
              "google_scholar": content_data.get("intermediate_google_scholar_agent", ""),
              "perplexity": content_data.get("intermediate_perplexity_agent", ""),
              "news": content_data.get("intermediate_news_agent", ""),
              "writer": content_data.get("intermediate_writer_agent", ""),
          }
      }
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# API Endpoint to fetch final feedback and resume the human input task
# -------------------------------
@app.post("/final-feedback")
async def final_feedback(request: FinalFeedbackRequest, user_data: dict = Depends(verify_firebase_token)):
  try:
      user_id = user_data["uid"]
      
      crew = pending_crews.get(request.reportId)
      if crew is None:
          raise HTTPException(status_code=404, detail="Report not found or already finalized.")

      # Use the human feedback provided to resume the editing task
      final_content = crew.provide_human_input(request.feedback)

      # Remove the crew reference as it is no longer pending
      del pending_crews[request.reportId]
      
      # Update the content document in Firestore
      content_query = db.collection("content").where("reportId", "==", request.reportId).limit(1)
      content_docs = content_query.stream()
      
      for doc in content_docs:
          content_ref = db.collection("content").document(doc.id)
          content_ref.update({
              "finalContent": final_content,
              "feedback": request.feedback,
              "status": "completed",
              "completedAt": firestore.SERVER_TIMESTAMP
          })
          
          return {
              "final_content": final_content,
              "contentId": doc.id
          }
      
      raise HTTPException(status_code=404, detail="Content document not found")
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# API Endpoint to get user data
# -------------------------------
@app.get("/user")
async def get_user(user_data: dict = Depends(verify_firebase_token)):
  try:
      user_id = user_data["uid"]
      user_ref = db.collection("users").document(user_id)
      user_doc = user_ref.get()
      
      if not user_doc.exists:
          raise HTTPException(status_code=404, detail="User not found")
      
      user_data = user_doc.to_dict()
      
      # Get content count
      content_query = db.collection("content").where("userId", "==", user_id)
      content_count = len(list(content_query.stream()))
      
      user_data["contentCount"] = content_count
      
      return user_data
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# API Endpoint to get user content
# -------------------------------
@app.get("/content")
async def get_content(user_data: dict = Depends(verify_firebase_token)):
  try:
      user_id = user_data["uid"]
      content_query = db.collection("content").where("userId", "==", user_id).order_by("createdAt", direction=firestore.Query.DESCENDING)
      content_docs = content_query.stream()
      
      content_list = []
      for doc in content_docs:
          content_data = doc.to_dict()
          content_data["id"] = doc.id
          
          # Convert Firestore timestamps to ISO format strings
          if "createdAt" in content_data and content_data["createdAt"]:
              content_data["createdAt"] = content_data["createdAt"].isoformat()
          if "completedAt" in content_data and content_data["completedAt"]:
              content_data["completedAt"] = content_data["completedAt"].isoformat()
              
          content_list.append(content_data)
      
      return content_list
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# API Endpoint to get specific content
# -------------------------------
@app.get("/content/{content_id}")
async def get_content_by_id(content_id: str, user_data: dict = Depends(verify_firebase_token)):
  try:
      user_id = user_data["uid"]
      content_ref = db.collection("content").document(content_id)
      content_doc = content_ref.get()
      
      if not content_doc.exists:
          raise HTTPException(status_code=404, detail="Content not found")
      
      content_data = content_doc.to_dict()
      
      # Verify this content belongs to the current user
      if content_data["userId"] != user_id:
          raise HTTPException(status_code=403, detail="You don't have permission to access this content")
      
      content_data["id"] = content_id
      
      # Convert Firestore timestamps to ISO format strings
      if "createdAt" in content_data and content_data["createdAt"]:
          content_data["createdAt"] = content_data["createdAt"].isoformat()
      if "completedAt" in content_data and content_data["completedAt"]:
          content_data["completedAt"] = content_data["completedAt"].isoformat()
      
      return content_data
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# API Endpoint to delete content
# -------------------------------
@app.delete("/content/{content_id}")
async def delete_content(content_id: str, user_data: dict = Depends(verify_firebase_token)):
  try:
      user_id = user_data["uid"]
      content_ref = db.collection("content").document(content_id)
      content_doc = content_ref.get()
      
      if not content_doc.exists:
          raise HTTPException(status_code=404, detail="Content not found")
      
      content_data = content_doc.to_dict()
      
      # Verify this content belongs to the current user
      if content_data["userId"] != user_id:
          raise HTTPException(status_code=403, detail="You don't have permission to delete this content")
      
      # Delete the content
      content_ref.delete()
      
      return {"message": "Content deleted successfully"}
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# API Endpoint to update user plan
# -------------------------------
@app.post("/update-plan")
async def update_plan(plan_data: dict, user_data: dict = Depends(verify_firebase_token)):
  try:
      user_id = user_data["uid"]
      new_plan = plan_data.get("plan")
      
      if not new_plan or new_plan not in ["starter", "professional", "enterprise"]:
          raise HTTPException(status_code=400, detail="Invalid plan specified")
      
      # Calculate new word allowance based on plan
      words_allowance = 10000 if new_plan == "starter" else 50000 if new_plan == "professional" else 200000
      
      # Update user's plan in Firestore
      user_ref = db.collection("users").document(user_id)
      user_ref.update({
          "plan": new_plan,
          "wordsRemaining": words_allowance,
          "planUpdatedAt": firestore.SERVER_TIMESTAMP
      })
      
      return {
          "message": f"Plan updated to {new_plan}",
          "plan": new_plan,
          "wordsRemaining": words_allowance
      }
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Run the application
# -------------------------------
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
