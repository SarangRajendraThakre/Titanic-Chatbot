from fastapi import FastAPI, Response, HTTPException
import pandas as pd
from pydantic import BaseModel
from langchain_openai import ChatOpenAI  # Updated import
from langchain_experimental.agents import create_pandas_dataframe_agent
import os
import io
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fastapi.responses import FileResponse

# Load environment variables
load_dotenv()

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY as an environment variable.")

# Load dataset
try:
    df = pd.read_csv("titanic.csv")
except FileNotFoundError:
    raise FileNotFoundError("Dataset file 'titanic.csv' not found. Please check the file path.")

# Initialize FastAPI
app = FastAPI()

# Setup LangChain Agent (Removed allow_dangerous_code=True for security)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
agent = create_pandas_dataframe_agent(llm, df, verbose=True, enable_code_execution=True, allow_dangerous_code=True)


# Define input format
class QueryRequest(BaseModel):
    question: str

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

            # Save the plot
            img_path = "histogram.png"
            plt.savefig(img_path)
            plt.close()

            # Return the image as response
            return FileResponse(img_path, media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

    else:
        try:
            # Normal text-based processing
            result = agent.run(question)
            return {"answer": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
