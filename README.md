﻿
# Titanic-Chatbot
![image](https://github.com/user-attachments/assets/5ebb2081-2891-477b-8d3f-c541ce53726e)


# Titanic-Chatbot  the dataset I used 
https://www.kaggle.com/datasets/yasserh/titanic-dataset/data

## Project Setup Guide

This guide will walk you through setting up and running the Titanic-Chatbot project. The project consists of a backend built using FastAPI and a frontend built with Streamlit. Follow these steps to get everything up and running.

---

### Clone the Project

First, clone the repository to your local machine:

```bash
git clone https://github.com/SarangRajendraThakre/Titanic-Chatbot.git
```

Change to the project directory:

```bash
cd Titanic-Chatbot
```

---

### Setup Backend

1. Change to the `backend` directory:

   ```bash
   cd backend
   ```

2. Create a `.env` file with your OpenAI API key:

   ```bash
   echo 'OPENAI_API_KEY="your_openai_api_key"' > .env
   ```

---

### Create a Virtual Environment

1. Create a Python virtual environment:

   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:

   - **Windows (CMD/PowerShell):**

     ```bash
     venv\Scriptsctivate
     ```

   - **Mac/Linux:**

     ```bash
     source venv/bin/activate
     ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

### Run the Backend

1. Change to the `backend` directory (if you aren't already there):

   ```bash
   cd backend
   ```

2. Run the FastAPI backend with Uvicorn:

   ```bash
   uvicorn main:app --reload
   ```

   This will start the backend server locally, and you should be able to access it at `http://127.0.0.1:8000`.

---

### Run the Frontend

1. Change to the `frontend` directory:

   ```bash
   cd ../frontend
   ```

2. Run the Streamlit frontend:

   ```bash
   streamlit run app.py
   ```

   This will start the Streamlit app in your browser.

---



### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute or reach out for any issues. Happy coding! 😊
