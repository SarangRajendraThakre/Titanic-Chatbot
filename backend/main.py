from fastapi import FastAPI, Response, HTTPException
import pandas as pd
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import os
import io
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fastapi.responses import FileResponse

# Load environment variables from a .env file
load_dotenv()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY as an environment variable.")

# Define file path for the dataset (use the relative path if deploying on a cloud)
dataset_path = os.path.join(os.path.dirname(__file__), 'titanic.csv')

# Load dataset
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found. Please check the file path.")

# Initialize FastAPI
app = FastAPI()

# Setup LangChain Agent
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
agent = create_pandas_dataframe_agent(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY),
    dataframe=df,
    verbose=True,
    enable_code_execution=True,
    allow_dangerous_code=True
)
# Define input format for query requests
class QueryRequest(BaseModel):
    question: str

# Define POST endpoint to process queries
@app.post("/query/")
async def query_data(request: QueryRequest):
    question = request.question.lower()  # Convert to lowercase for better matching
    
    # Check if the user wants a visualization
    if any(word in question for word in ["histogram", "plot", "graph", "chart"]):
        try:
            # Create histogram for passenger ages
            plt.figure(figsize=(8, 5))
            plt.hist(df["Age"].dropna(), bins=20, color='skyblue', edgecolor='black')
            plt.xlabel("Age")
            plt.ylabel("Number of Passengers")
            plt.title("Histogram of Passenger Ages")
            plt.grid(True)

            # Save the plot to an image file
            img_path = "histogram.png"
            plt.savefig(img_path)
            plt.close()

            # Return the image as a response
            return FileResponse(img_path, media_type="image/png")
        except Exception as e:
            # Catching any issues that occur during plot generation
            raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

    else:
        try:
            # Normal text-based processing
            result = agent.run(question)
            return {"answer": result}
        except Exception as e:
            # Catching any issues that occur during query processing
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
