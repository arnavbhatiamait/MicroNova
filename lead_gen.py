import os
import io
import tempfile
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
import google.generativeai as genai
import ollama
import asyncio
import subprocess
import base64
import pywhatkit
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from PIL import Image
import base64
import io
import os
# Set matplotlib backend
import matplotlib
from langchain_core.messages import HumanMessage
import requests
matplotlib.use('Agg')
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq 
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def send_whats_message(message,no):
    pywhatkit.sendwhatmsg_instantly(f"+91{no}", message, 10,False, 5)
def get_ollama_models():
    """Fetches a list of available Ollama models."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        models = []
        for line in lines[1:]: # Skip header row
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except subprocess.CalledProcessError as e:
        st.error(f"Error running 'ollama list': {e.stderr}")
        return []
    except FileNotFoundError:
        st.error("Ollama command not found. Please ensure Ollama is installed and in your system's PATH.")
        return []

def get_all_gemini_models(api_key):
    """
    Lists the names of all available Gemini models.

    Args:
        api_key: Your Google AI API key.

    Returns:
        A list of strings, where each string is the name of an available Gemini model.
        Returns an empty list if the API key is invalid or there's an error.
    """
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        gemini_models = [model.name for model in models if "gemini" in model.name.lower() and "generateContent" in model.supported_generation_methods]
        return gemini_models
    except Exception as e:
        st.error(f"An error occurred while fetching Gemini models: {e}")
        return []

def get_all_groq_models(api_key: str) -> list:
    """
    Fetches and returns a list of all model IDs available from the Groq API.

    Args:
        api_key (str): Your Groq API key.

    Returns:
        List[str]: A list of model IDs (names).
    """
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        models_data = response.json()
        return [model["id"] for model in models_data.get("data", [])]
    except requests.RequestException as e:
        print(f"Error fetching Groq models: {e}")
        return []

# --- Streamlit UI for Model Selection ---
st.set_page_config(page_title="Vistas",page_icon="./Vistas_1.png")

st.image("./Vistas_1.png") # Assuming you have this image

with st.sidebar:
    st.title("Choose Model Parameters")
    model_choice = st.selectbox("Select Model", ["Gemini", "Ollama","Groq","OpenAI"])
    GEMINI_AVAILABLE = False
    OLLAMA_AVAILABLE = False
    GROQ_AVAILABLE = False
    OPENAI_AVAILABLE = False
    model_name = ""

    if model_choice == "Gemini":
        api_key = st.text_input("Enter Gemini API Key", type='password')
        if api_key:
            model_list = get_all_gemini_models(api_key=api_key)
            if model_list:
                model_name = st.selectbox("Choose Gemini Model Name", model_list, index=model_list.index('gemini-1.5-flash-latest') if 'gemini-1.5-flash-latest' in model_list else 0)
                genai.configure(api_key=api_key)
                GEMINI_AVAILABLE = True
                st.success(f"{model_name} is available.")
            else:
                st.warning("No Gemini models found or API key is invalid.")
        else:
            st.warning("Please enter your Gemini API Key.")
    elif model_choice=="Ollama": # Llama
        model_list = get_ollama_models()
        if model_list:
            model_name = st.selectbox("Choose Ollama Model Name", model_list)
            OLLAMA_AVAILABLE = True
            st.success(f"{model_name} is available.")
        else:
            st.warning("No Ollama models found. Please ensure Ollama is running and models are pulled.")
    elif model_choice=="Groq": 
        api_key = st.text_input("Enter GROQ API Key", type='password')

        # api_key = st.text_input("Enter Groq API Key", type='password')
        if api_key:
            model_list = get_all_groq_models(api_key=api_key)

            if model_list:
                model_name = st.selectbox("Choose Groq Model Name", model_list)
                GROQ_AVAILABLE = True
                st.success(f"{model_name} is available.")
            else:
                st.warning("No GROQ models found")
        else:
            st.warning("Please enter your Groq API Key.")
    elif model_choice=="OpenAI": 
        model_list = []
        api_key = st.text_input("Enter OpenAI API Key", type='password')
        if api_key:
            # model_list = get_all_gemini_models(api_key=api_key)
            model_list = [
                # GPT-4.1 series
                "gpt-4.1",
                "gpt-4.1-mini",
                "gpt-4.1-nano",

                # GPT-4o ("Omni") series
                "gpt-4o",
                "gpt-4o-mini",

                # GPT-4.5 ("Orion")
                "gpt-4.5",

                # Reasoning-focused models (o-series)
                "o3",
                "o3-mini",
                "o4-mini",
                "o4-mini-high",

                # Image generation models
                "dall-e-3"
            ]

            if model_list:
                model_name = st.selectbox("Choose OpenAI Model Name", model_list)
                OPENAI_AVAILABLE = True
                st.success(f"{model_name} is available.")
            else:
                st.warning("No OpenAI  models found")
        else:
            st.warning("Please enter your OpenAI API Key.")

    st.title("Send WhatsApp Message")
    message = st.text_input("Enter WhatsApp Message")
    no=st.text_input("Enter WhatsApp Number (without +91)")
    if st.button("Send WhatsApp Message"):
        if message and no:
            try:
                send_whats_message(message,no)
                st.success("WhatsApp message sent successfully!")
            except Exception as e:
                st.error(f"Error sending WhatsApp message: {e}")
        else:
            st.warning("Please enter both message and number.")


# Store selected model globally for use in functions
GEMINI_MODEL = model_name if model_choice == "Gemini" else None
LLAMA_MODEL = model_name if model_choice == "Ollama" else None
GROQ_MODEL= model_name if model_choice == "Groq" else None
OPENAI_MODEL = model_name if model_choice == "OpenAI" else None

# Helper functions
def save_fig(fig):
    """Saves a matplotlib figure to a temporary PNG file and returns the path."""
    f = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fig.savefig(f.name, bbox_inches='tight')
    plt.close(fig)
    return f.name

def df_into_string(df, max_rows=5):
    """
    Generates a string summary of a DataFrame including schema, head, and missing values.
    """
    buf = io.StringIO()
    df.info(buf=buf)
    schema = buf.getvalue()
    head = df.head(max_rows).to_markdown(index=False)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_info = "No missing values" if missing.empty else f"Missing values: {missing.to_dict()}"
    return f"### Schema:\n```\n{schema}```\n### Preview:\n```\n{head}```\n### {missing_info}"

async def run_gemini_analysis(prompt_key, df_context):
    """Runs text analysis using the selected Gemini model."""
    if not GEMINI_AVAILABLE or not GEMINI_MODEL:
        return "Gemini model is not configured or available."
    
    
    prompts = {
        
        "best_leads": f"List the top 5 customers based on credit card, affordabilty classify bassed on the salary,credit score, type of work, emplyment status,age using that generate a lead score from 0 to 5 and return the following  Sno, Name,email_id, age, credit_score, employment status, lead score, phone no order by highest lead score from 5 to 3:\n{df_context}, Do not give any code and provide output in tabular format only no other explanation text needed", 
        "worst_leads": f"List the worst 5 customers based on credit card affordabilty classify bassed on the salary,credit score, type of work, emplyment status,age using that generate a lead score from 0 to 5 and return the following Sno, Name,email_id, age, credit_score, employment status, lead score, phone no order by lowest lead score from 0 to 2:\n{df_context}, Do not give any code and provide output in tabular format only no other explanation text needed",
        "future": f"Suggest future customers and strategies for a Credit card company based on this data classify bassed on the salary,credit score, type of work, emplyment status,age using that generate a lead score from 0 to 5 and return the following Sno, Name,email_id, age, credit_score, employment status, lead score, phone no : :\n{df_context} ,Do not give any code and provide output in tabular format only no other explanation text needed and add a column for explanation for keeping that customer",
        "lead_analysis": f"Analyze the following customer/lead data. Provide insights into potential high-value leads, common lead sources, and suggest strategies for nurturing different lead segments. Consider customer names, contact info, past interactions, demographic data, and firmographic data.\n{df_context}"
    }
    

    
    try:
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        res = await model.generate_content_async(
            prompts.get(prompt_key),
            generation_config=genai.types.GenerationConfig(max_output_tokens=1500, temperature=0.3)
        )
        return res.text if res.parts else "No response from Gemini model."
    except Exception as e:
        return f"Error with Gemini model: {e}"

async def run_ollama_analysis(prompt_key, df_context):
    """Runs text analysis using the selected Ollama model."""
    if not OLLAMA_AVAILABLE or not LLAMA_MODEL:
        return "Ollama model is not configured or available."

    prompts = {
        
        "best_leads": f"List the top 5 customers based on credit card, affordabilty classify bassed on the salary,credit score, type of work, emplyment status,age using that generate a lead score from 0 to 5 and return the following  Sno, Name,email_id, age, credit_score, employment status, lead score, phone no order by highest lead score from 5 to 3:\n{df_context}, Do not give any code and provide output in tabular format only no other explanation text needed", 
        "worst_leads": f"List the worst 5 customers based on credit card affordabilty classify bassed on the salary,credit score, type of work, emplyment status,age using that generate a lead score from 0 to 5 and return the following Sno, Name,email_id, age, credit_score, employment status, lead score, phone no order by lowest lead score from 0 to 2:\n{df_context}, Do not give any code and provide output in tabular format only no other explanation text needed",
        "future": f"Suggest future customers and strategies for a Credit card company based on this data classify bassed on the salary,credit score, type of work, emplyment status,age using that generate a lead score from 0 to 5 and return the following Sno, Name,email_id, age, credit_score, employment status, lead score, phone no : :\n{df_context} ,Do not give any code and provide output in tabular format only no other explanation text needed and add a column for explanation for keeping that customer",
        "lead_analysis": f"Analyze the following customer/lead data. Provide insights into potential high-value leads, common lead sources, and suggest strategies for nurturing different lead segments. Consider customer names, contact info, past interactions, demographic data, and firmographic data.\n{df_context}"
    }
    

    try:
        res = ollama.generate(prompt=prompts.get(prompt_key), model=LLAMA_MODEL)
        return res.get("response", "No response from Ollama model.")
    except Exception as e:
        return f"Error with Ollama model: {e}"

async def run_groq_analysis(prompt_key, df_context):
    """Runs text analysis using the selected Groq model."""
    if not GROQ_AVAILABLE or not GROQ_MODEL:
        return "GROQ model is not configured or available."

    prompts = {
        
        "best_leads": f"List the top 5 customers based on credit card, affordabilty classify bassed on the salary,credit score, type of work, emplyment status,age using that generate a lead score from 0 to 5 and return the following  Sno, Name,email_id, age, credit_score, employment status, lead score, phone no order by highest lead score from 5 to 3:\n{df_context}, Do not give any code and provide output in tabular format only no other explanation text needed", 
        "worst_leads": f"List the worst 5 customers based on credit card affordabilty classify bassed on the salary,credit score, type of work, emplyment status,age using that generate a lead score from 0 to 5 and return the following Sno, Name,email_id, age, credit_score, employment status, lead score, phone no order by lowest lead score from 0 to 2:\n{df_context}, Do not give any code and provide output in tabular format only no other explanation text needed",
        "future": f"Suggest future customers and strategies for a Credit card company based on this data classify bassed on the salary,credit score, type of work, emplyment status,age using that generate a lead score from 0 to 5 and return the following Sno, Name,email_id, age, credit_score, employment status, lead score, phone no : :\n{df_context} ,Do not give any code and provide output in tabular format only no other explanation text needed and add a column for explanation for keeping that customer",
        "lead_analysis": f"Analyze the following customer/lead data. Provide insights into potential high-value leads, common lead sources, and suggest strategies for nurturing different lead segments. Consider customer names, contact info, past interactions, demographic data, and firmographic data.\n{df_context}"
    }
    
    prompt= prompts.get(prompt_key)
    llm = ChatGroq(model=GROQ_MODEL, api_key=api_key)
    try:
        # res = ollama.generate(prompt=prompts.get(prompt_key), model=LLAMA_MODEL).
        res = llm.invoke([HumanMessage(prompt)]).content.strip()
        return res if res else "No response from Groq model."
    except Exception as e:
        return f"Error with Groq model: {e}"

async def run_openai_analysis(prompt_key, df_context):
    """Runs text analysis using the selected OPENAI model."""
    if not OPENAI_AVAILABLE or not OPENAI_MODEL:
        return "GROQ model is not configured or available."

    prompts = {
        
        "best_leads": f"List the top 5 customers based on credit card, affordabilty classify bassed on the salary,credit score, type of work, emplyment status,age using that generate a lead score from 0 to 5 and return the following  Sno, Name,email_id, age, credit_score, employment status, lead score, phone no order by highest lead score from 5 to 3:\n{df_context}, Do not give any code and provide output in tabular format only no other explanation text needed", 
        "worst_leads": f"List the worst 5 customers based on credit card affordabilty classify bassed on the salary,credit score, type of work, emplyment status,age using that generate a lead score from 0 to 5 and return the following Sno, Name,email_id, age, credit_score, employment status, lead score, phone no order by lowest lead score from 0 to 2:\n{df_context}, Do not give any code and provide output in tabular format only no other explanation text needed",
        "future": f"Suggest future customers and strategies for a Credit card company based on this data classify bassed on the salary,credit score, type of work, emplyment status,age using that generate a lead score from 0 to 5 and return the following Sno, Name,email_id, age, credit_score, employment status, lead score, phone no : :\n{df_context} ,Do not give any code and provide output in tabular format only no other explanation text needed and add a column for explanation for keeping that customer",
        "lead_analysis": f"Analyze the following customer/lead data. Provide insights into potential high-value leads, common lead sources, and suggest strategies for nurturing different lead segments. Consider customer names, contact info, past interactions, demographic data, and firmographic data.\n{df_context}"
    }
    
    prompt= prompts.get(prompt_key)
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=api_key)
    try:
        # res = ollama.generate(prompt=prompts.get(prompt_key), model=LLAMA_MODEL).
        res = llm.invoke(prompt=[HumanMessage(content=prompt)]).content.strip()
        return res if res else "No response from OpenAI model."
    except Exception as e:
        return f"Error with OpenAI model: {e}"



# def visual_generation(df):
#     """
#     Generates relevant visualizations from the DataFrame.
#     Extended to include lead-specific visualizations.
#     """
#     visualizations = []
    
#     try:
#         # --- General Data Visualizations (from original code) ---
#         if 'date' in df.columns:
#             df['date'] = pd.to_datetime(df['date'], errors='coerce')
#             df_time = df.dropna(subset=['date'])
#             if not df_time.empty:
#                 sales_over_time = df_time.groupby('date')['loan_amount'].sum() if 'loan_amount' in df.columns else None
#                 if sales_over_time is not None:
#                     fig, ax = plt.subplots(figsize=(10,6))
#                     sales_over_time.plot(ax=ax)
#                     ax.set_title("Total Loan Amount Over Time")
#                     path = save_fig(fig)
#                     visualizations.append(("Loan Amount Over Time", path))
        
#         if 'repayment_status' in df.columns:
#             status_counts = df['repayment_status'].value_counts()
#             if not status_counts.empty:
#                 fig, ax = plt.subplots()
#                 sns.barplot(x=status_counts.index, y=status_counts.values, ax=ax)
#                 ax.set_title("Repayment Status Distribution")
#                 path = save_fig(fig)
#                 visualizations.append(("Repayment Status Distribution", path))

#         if 'loan_amount' in df.columns:
#             fig, ax = plt.subplots()
#             sns.histplot(df['loan_amount'], kde=True, ax=ax)
#             ax.set_title("Loan Amount Distribution")
#             path = save_fig(fig)
#             visualizations.append(("Loan Amount Distribution", path))
        
#         if 'customer_id' in df.columns and 'loan_amount' in df.columns:
#             top_customers = df.groupby('customer_id')['loan_amount'].sum().sort_values(ascending=False).head(10)
#             if not top_customers.empty:
#                 fig, ax = plt.subplots()
#                 top_customers.plot(kind='bar', ax=ax)
#                 ax.set_title("Top 10 Customers by Loan Amount")
#                 path = save_fig(fig)
#                 visualizations.append(("Top Customers", path))

#         if 'Order Date' in df.columns:
#             df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
#             df_time = df.dropna(subset=['Order Date']).copy()
#             if not df_time.empty and 'Total Revenue' in df.columns:
#                 sales_over_time = df_time.groupby('Order Date')['Total Revenue'].sum()
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 sales_over_time.plot(ax=ax)
#                 ax.set_title("Total Revenue Over Time")
#                 ax.set_xlabel("Date")
#                 ax.set_ylabel("Total Revenue")
#                 path = save_fig(fig)
#                 visualizations.append(("Total Revenue Over Time", path))

#         if 'Item Type' in df.columns and 'Total Revenue' in df.columns:
#             item_sales = df.groupby('Item Type')['Total Revenue'].sum().sort_values(ascending=False)
#             if not item_sales.empty:
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 item_sales.plot(kind='bar', ax=ax)
#                 ax.set_title("Total Revenue by Item Type")
#                 ax.set_xlabel("Item Type")
#                 ax.set_ylabel("Total Revenue")
#                 path = save_fig(fig)
#                 visualizations.append(("Total Revenue by Item Type", path))

#         if 'Region' in df.columns and 'Total Revenue' in df.columns:
#             region_sales = df.groupby('Region')['Total Revenue'].sum().sort_values(ascending=False)
#             if not region_sales.empty:
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 region_sales.plot(kind='bar', ax=ax)
#                 ax.set_title("Total Revenue by Region")
#                 ax.set_xlabel("Region")
#                 ax.set_ylabel("Total Revenue")
#                 path = save_fig(fig)
#                 visualizations.append(("Total Revenue by Region", path))

#         if 'Order Priority' in df.columns:
#             priority_counts = df['Order Priority'].value_counts()
#             if not priority_counts.empty:
#                 fig, ax = plt.subplots()
#                 sns.barplot(x=priority_counts.index, y=priority_counts.values, ax=ax)
#                 ax.set_title("Order Priority Distribution")
#                 ax.set_xlabel("Order Priority")
#                 ax.set_ylabel("Count")
#                 path = save_fig(fig)
#                 visualizations.append(("Order Priority Distribution", path))

#         if 'Total Profit' in df.columns and 'Total Revenue' in df.columns:
#             fig, ax = plt.subplots()
#             sns.scatterplot(x='Total Revenue', y='Total Profit', data=df, ax=ax)
#             ax.set_title("Total Profit vs Total Revenue")
#             ax.set_xlabel("Total Revenue")
#             ax.set_ylabel("Total Profit")
#             path = save_fig(fig)
#             visualizations.append(("Total Profit vs Total Revenue", path))

#         # --- Lead Generation Specific Visualizations ---
#         if 'Lead Source' in df.columns: # Assuming a 'Lead Source' column for incoming leads
#             lead_source_counts = df['Lead Source'].value_counts()
#             if not lead_source_counts.empty:
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 sns.barplot(x=lead_source_counts.index, y=lead_source_counts.values, ax=ax)
#                 ax.set_title("Leads by Source")
#                 ax.set_xlabel("Lead Source")
#                 ax.set_ylabel("Number of Leads")
#                 ax.tick_params(axis='x', rotation=45)
#                 path = save_fig(fig)
#                 visualizations.append(("Leads by Source", path))

#         if 'Industry' in df.columns: # Assuming an 'Industry' column for firmographic data
#             industry_counts = df['Industry'].value_counts().head(10) # Top 10 industries
#             if not industry_counts.empty:
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 sns.barplot(x=industry_counts.index, y=industry_counts.values, ax=ax)
#                 ax.set_title("Top 10 Leads by Industry")
#                 ax.set_xlabel("Industry")
#                 ax.set_ylabel("Number of Leads")
#                 ax.tick_params(axis='x', rotation=45)
#                 path = save_fig(fig)
#                 visualizations.append(("Leads by Industry", path))
        
#         if 'Interaction Type' in df.columns: # Assuming 'Interaction Type' (e.g., Email, Call, Meeting)
#             interaction_counts = df['Interaction Type'].value_counts()
#             if not interaction_counts.empty:
#                 fig, ax = plt.subplots(figsize=(8, 8))
#                 ax.pie(interaction_counts, labels=interaction_counts.index, autopct='%1.1f%%', startangle=90)
#                 ax.set_title("Distribution of Interaction Types")
#                 ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
#                 path = save_fig(fig)
#                 visualizations.append(("Interaction Type Distribution", path))

#     except Exception as e:
#         st.error(f"Error generating visualizations: {e}")
#         traceback.print_exc() # Print full traceback for debugging
#     return visualizations


import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
def save_fig(fig):
    """Saves a matplotlib figure to a temporary PNG file and returns the path."""
    f = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fig.savefig(f.name, bbox_inches='tight')
    plt.close(fig)
    return f.name
def visual_generation(df):
    visualizations = []

    try:
        cols = df.columns.str.lower().str.replace(" ", "_")
        df.columns = cols  # Standardize column names

        if 'age' in cols:
            fig, ax = plt.subplots()
            sns.histplot(df['age'].dropna(), bins=20, kde=True, ax=ax,color="blue")
            ax.set_title("Age Distribution")
            path = save_fig(fig)
            visualizations.append(("Age Distribution", path))

        if 'salary' in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df['salary'].dropna(), kde=True, ax=ax,color="orange")
            ax.set_title("Salary Distribution")
            path = save_fig(fig)
            visualizations.append(("Salary Distribution", path))

        if 'credit_score' in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df['credit_score'].dropna(), kde=True, ax=ax,color="green")
            ax.set_title("Credit Score Distribution")
            path = save_fig(fig)
            visualizations.append(("Credit Score Distribution", path))

        if 'salary' in df.columns and 'credit_score' in df.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(x='salary', y='credit_score', data=df, ax=ax, color="red")
            ax.set_title("Salary vs Credit Score")
            path = save_fig(fig)
            visualizations.append(("Salary vs Credit Score", path))

        if 'employment_status' in df.columns and 'salary' in df.columns:
            salary_by_status = df.groupby('employment_status')['salary'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots()
            salary_by_status.plot(kind='bar', ax=ax, color='pink')
            ax.set_title("Average Salary by Employment Status")
            path = save_fig(fig)
            visualizations.append(("Average Salary by Employment Status", path))

        if 'employment_status' in cols and 'credit_score' in cols:
            fig, ax = plt.subplots()
            sns.boxplot(x='employment_status', y='credit_score', data=df, ax=ax,palette="mako")
            ax.set_title("Employment Status vs Credit Score")
            ax.tick_params(axis='x', rotation=45)
            path = save_fig(fig)
            visualizations.append(("Employment vs Credit Score", path))

        # Add more graphs as before...

    except Exception as e:
        # st.error(f"Error generating visualizations: {e}")
        # traceback.print_exc()

        print(f"Error generating visualizations: {e}")
    return visualizations

def cleanup(files):
    """Deletes temporary files."""
    for file in files:
        try:
            os.remove(file)
        except Exception:
            pass

# async def run_gemini_vision_analysis(image_path_list):
#     """Analyzes images using Gemini Vision model."""
#     if not GEMINI_AVAILABLE or not GEMINI_MODEL:
#         return [{"title": "AI Vision", "response": "Gemini model is not configured or available."}]
    
#     # Ensure a vision-capable model is selected
#     # try:
#     #     model_info = genai.get_model(GEMINI_MODEL)
#     #     if "VISION" not in model_info.supported_generation_methods:
#     #          st.warning(f"Selected Gemini model '{GEMINI_MODEL}' does not support vision. Please choose a vision-capable model (e.g., gemini-1.5-flash-latest).")
#     #          return [{"title": "AI Vision", "response": "Selected Gemini model does not support vision."}]
#     # except Exception as e:
#     #     st.error(f"Could not get model info for '{GEMINI_MODEL}': {e}")
#     #     return [{"title": "AI Vision", "response": f"Error getting model info: {e}"}]

#     model = genai.GenerativeModel(model_name=GEMINI_MODEL)
#     results = []
#     for title, path in image_path_list:
#         try:
#             img = Image.open(path)
#             res = await model.generate_content_async(
#                 [f"Explain this chart titled '{title}' in the context of the data it represents.", img],
#                 generation_config=genai.types.GenerationConfig(max_output_tokens=1000, temperature=0.2)
#             )
#             results.append({"title": title, "response": res.text if res.parts else "No response from Gemini model."})
#         except Exception as e:
#             results.append({"title": title, "response": f"Error during Gemini vision analysis for '{title}': {e}"})
#     return results
async def run_gemini_vision_analysis(image_path):
    if not GEMINI_AVAILABLE:
        return [{"AI vision","Gemini not available"}] 
    model=genai.GenerativeModel(model_name=GEMINI_MODEL)
    results=[]
    for title,path in image_path:
        try:
            img=Image.open(path)
            res=await model.generate_content_async([f"Explain this '{title}'",img],generation_config=genai.types.GenerationConfig(max_output_tokens=1000,temperature=0.2))
            results.append({"title":title,"response":res.text if res.parts else "No response from Gemini model"})
        except Exception as e:
            results.append({"title":title,"response":f"Error: {e}"})
    return results

    
async def run_ollama_vision_analysis(image_path_list):
    """Analyzes images using Ollama (Llama) Vision model."""
    results = []
    if not OLLAMA_AVAILABLE or not LLAMA_MODEL:
        return [{"title": "AI Vision", "response": "Ollama model is not configured or available."}]

    for title, path in image_path_list:
        try:
            img = Image.open(path)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Ollama's generate expects a direct call, not an async one, in this context
            response = ollama.generate(
                model=LLAMA_MODEL,
                prompt=f"Explain this chart titled '{title}' in the context of the data it represents.",
                images=[img_str],
                options={"temperature": 0.2, "num_predict": 1000}
            )
            results.append({
                "title": title,
                "response": response.get("response", "No response from Ollama model.")
            })
        except Exception as e:
            results.append({
                "title": title,
                "response": f"Error during Ollama vision analysis for '{title}': {e}"
            })
    return results



# Set this to your actual model if Groq supports multimodal, otherwise use gpt-4o

async def run_groq_vision_analysis(image_path_list):
    """Analyzes images using a LangChain-compatible Groq/OpenAI vision model."""
    results = []
    if not GROQ_AVAILABLE or not GROQ_MODEL:
        return [{"title": "AI Vision", "response": "Ollama model is not configured or available."}]
    LLM_VISION = ChatGroq(model=GROQ_MODEL, temperature=0.2)


    for title, path in image_path_list:
        try:
            # Convert image to base64 format
            img = Image.open(path)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Construct the vision message
            message = HumanMessage(
                content=[
                    {"type": "text", "text": f"Explain this chart titled '{title}' in the context of the data it represents."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            )

            # Run async invocation
            response = await LLM_VISION.invoke([message])
            results.append({
                "title": title,
                "response": response.content.strip()
            })

        except Exception as e:
            results.append({
                "title": title,
                "response": f"Error during vision analysis for '{title}': {e}"
            })

    return results


async def run_openai_vision_analysis(image_path_list):
    """Analyzes images using a LangChain-compatible Groq/OpenAI vision model."""
    results = []
    if not OPENAI_AVAILABLE or not OPENAI_MODEL:
        return [{"title": "AI Vision", "response": "Ollama model is not configured or available."}]
    LLM_VISION = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)


    for title, path in image_path_list:
        try:
            # Convert image to base64 format
            img = Image.open(path)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Construct the vision message
            message = HumanMessage(
                content=[
                    {"type": "text", "text": f"Explain this chart titled '{title}' in the context of the data it represents."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            )

            # Run async invocation
            response = await LLM_VISION.invoke([message])
            results.append({
                "title": title,
                "response": response.content.strip()
            })

        except Exception as e:
            results.append({
                "title": title,
                "response": f"Error during vision analysis for '{title}': {e}"
            })

    return results

# --- Streamlit App Main Logic ---
st.title("Microfinance & Lead Data Analysis Assistant")

uploaded_file = st.file_uploader("Upload your sales/lead data CSV file", type=["csv"])

if uploaded_file:
    # results={
    #     "top_leads": None,
    #     "worst_leads": "",
    #     "future_strategies": "",
    #     "lead_analysis": ""
    # }
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.warning("The uploaded file is empty.")
        else:
            st.success("File uploaded successfully!")
            st.write("Data Preview:")
            st.dataframe(df.head())

            # Show data info
            with io.StringIO() as buf:
                df.info(buf=buf)
                info_str_full = buf.getvalue()
            st.text_area("Data Schema & Info", value=info_str_full, height=200)
            # with tab_visual:
            # Generate visualizations
            st.subheader("Data Visualizations")
            visualizations = []
            try:
                visualizations = visual_generation(df)
                if visualizations:
                    for title, path in visualizations:
                        st.image(path, caption=title, use_container_width=True)
                else:
                    st.info("No standard visualizations could be generated from the available columns.")
            except Exception as e:
                st.error(f"Error generating visualizations: {e}")

            # Prepare info string for analysis
            df_context_string = df_into_string(df)

            st.subheader("Text-Based Analysis")

            # --- Microfinance/Sales Analysis Buttons ---
            st.markdown("#### Microfinance/Sales Insights")
            # col1, col2, col3 = st.columns(3)
            # with col1:
            #     # if st.button("Perform SWOT Analysis", key="swot_btn"):
            #         with st.spinner("Analyzing SWOT..."):
            #             if GEMINI_AVAILABLE:
            #                 result = asyncio.run(run_gemini_analysis("swot", df_context_string))
            #             elif OLLAMA_AVAILABLE:
            #                 result = asyncio.run(run_ollama_analysis("swot", df_context_string))
            #             elif GROQ_AVAILABLE:
            #                 result = asyncio.run(run_groq_analysis("swot", df_context_string))
            #             elif OPENAI_AVAILABLE:
            #                 result = asyncio.run(run_openai_analysis("swot", df_context_string))
            #             else:
            #                 result = "No model selected or available."
            #         st.markdown("### SWOT Analysis")
            #         st.write(result)
            tab_top, tab_worst, tab_future= st.tabs (["Top Leads", "Worst Leads", "Future Strategies"])

            with tab_top:
                if st.button("List Top Leads", key="top_leads_btn"):
                    with st.spinner("Listing top Leads..."):
                        if GEMINI_AVAILABLE:
                            result = asyncio.run(run_gemini_analysis("best_leads", df_context_string))
                        elif OLLAMA_AVAILABLE:
                            result = asyncio.run(run_ollama_analysis("best_leads", df_context_string))
                        elif GROQ_AVAILABLE:
                            result = asyncio.run(run_groq_analysis("best_leads", df_context_string))
                        elif OPENAI_AVAILABLE:
                            result = asyncio.run(run_openai_analysis("best_leads", df_context_string))
                        else:
                            result = "No model selected or available."
                    st.markdown("### Top Customers")
                    # results["top_leads"]=result
                    # print(results)
                    st.write(result)
                # if st.button("Send Whatsapp Message"):
                # no=st.text_input("Enter Phone Number for whatsapp Message")
                # if st.button("Send Message") and no:
                #     # if no:
                #     # with st.spinner("Sending message..."):
                #     try:
                #         # Assuming a function send_whatsapp_message exists
                #         print(results)
                #         pywhatkit.sendwhatmsg_instantly(f"+91{no}", results['top_leads'], 15, False, 5)

                #         st.success(f"Message sent to {no} successfully!")
                #     except Exception as e:
                #         st.error(f"Failed to send message: {e}")
                    # else:
                    #     st.warning("Please enter a valid phone number.")
            with tab_worst:
                if st.button("List Worst Customers", key="worst_leads_btn"):
                    with st.spinner("Listing worst customers..."):
                        if GEMINI_AVAILABLE:
                            result = asyncio.run(run_gemini_analysis("worst_leads", df_context_string))
                        elif OLLAMA_AVAILABLE:
                            result = asyncio.run(run_ollama_analysis("worst_leads", df_context_string))
                        elif GROQ_AVAILABLE:
                            result = asyncio.run(run_groq_analysis("worst_leads", df_context_string))
                        elif OPENAI_AVAILABLE:
                            result = asyncio.run(run_openai_analysis("worst_leads", df_context_string))
                        
                        else:
                            result = "No model selected or available."
                    st.markdown("### Worst Customers")
                    st.write(result)
                    # if st.button("Send Whatsapp Message"):
                    #     no=st.text_input("Enter Phone Number")
                    #     if st.button("Send Message"):
                    #         if no:
                    #             with st.spinner("Sending message..."):
                    #                 try:
                    #                     # Assuming a function send_whatsapp_message exists
                    #                     send_whats_message(no, result)
                    #                     st.success(f"Message sent to {no} successfully!")
                    #                 except Exception as e:
                    #                     st.error(f"Failed to send message: {e}")
                    #         else:
                    #             st.warning("Please enter a valid phone number.")
            with tab_future:
                if st.button("Suggest Future Strategies", key="future_strat_btn"):
                    with st.spinner("Suggesting strategies..."):
                        if GEMINI_AVAILABLE:
                            result = asyncio.run(run_gemini_analysis("future", df_context_string))
                        elif OLLAMA_AVAILABLE:
                            result = asyncio.run(run_ollama_analysis("future", df_context_string))
                        elif GROQ_AVAILABLE:
                            result = asyncio.run(run_groq_analysis("future", df_context_string))
                        elif OPENAI_AVAILABLE:
                            result = asyncio.run(run_openai_analysis("future", df_context_string))
                        else:
                            result = "No model selected or available."
                    st.markdown("### Future Strategies")
                    st.markdown(result)
                    # if st.button("Send Whatsapp Message"):
                    #     no=st.text_input("Enter Phone Number")
                    #     if st.button("Send Message"):
                    #         if no:
                    #             with st.spinner("Sending message..."):
                    #                 try:
                    #                     # Assuming a function send_whatsapp_message exists
                    #                     send_whats_message(no, result)
                    #                     st.success(f"Message sent to {no} successfully!")
                    #                 except Exception as e:
                    #                     st.error(f"Failed to send message: {e}")
                    #         else:
                    #             st.warning("Please enter a valid phone number.")

            # --- Lead Generation Analysis Button ---
            st.markdown("#### Lead Generation Insights")
            if st.button("Analyze Leads for Insights", key="lead_analysis_btn"):
                with st.spinner("Analyzing lead data..."):
                    if GEMINI_AVAILABLE:
                        lead_analysis_result = asyncio.run(run_gemini_analysis("lead_analysis", df_context_string))
                    elif OLLAMA_AVAILABLE:
                        lead_analysis_result = asyncio.run(run_ollama_analysis("lead_analysis", df_context_string))
                    elif GROQ_AVAILABLE:
                        lead_analysis_result = asyncio.run(run_groq_analysis("future", df_context_string))
                    elif OPENAI_AVAILABLE:
                        lead_analysis_result = asyncio.run(run_openai_analysis("future", df_context_string))
                    else:
                        lead_analysis_result = "No model selected or available for lead analysis."
                st.markdown("### Lead Analysis Insights")
                st.write(lead_analysis_result)
                # if st.button("Send Whatsapp Message"):
                #         no=st.text_input("Enter Phone Number")
                #         if st.button("Send Message"):
                #             if no:
                #                 with st.spinner("Sending message..."):
                #                     try:
                #                         # Assuming a function send_whatsapp_message exists
                #                         send_whats_message(no, lead_analysis_result)
                #                         st.success(f"Message sent to {no} successfully!")
                #                     except Exception as e:
                #                         st.error(f"Failed to send message: {e}")
                #             else:
                #                 st.warning("Please enter a valid phone number.")

            # --- Graph Description Button ---
            st.markdown("#### Visual Insights")
            if st.button("Describe Various Graphs", key="describe_graphs_btn"):
                if not visualizations:
                    st.warning("No visualizations were generated to describe.")
                else:
                    with st.spinner("Analyzing graphs with AI vision..."):
                        if GEMINI_AVAILABLE:
                            graph_descriptions = asyncio.run(run_gemini_vision_analysis(visualizations))
                        elif OLLAMA_AVAILABLE:
                            graph_descriptions = asyncio.run(run_ollama_vision_analysis(visualizations))
                        elif GROQ_AVAILABLE: 
                            graph_descriptions = asyncio.run(run_groq_vision_analysis(visualizations))
                        elif OPENAI_AVAILABLE:
                            graph_descriptions = asyncio.run(run_openai_vision_analysis(visualizations))
                        else:
                            graph_descriptions = [{"title": "AI Vision", "response": "No vision-capable model selected or available."}]
                        
                        st.markdown("### Graph Descriptions and Insights")
                        for item in graph_descriptions:
                            title = item.get("title")
                            insight = item.get("response")
                            st.markdown(f"**Insight for: {title}**")
                            st.write(insight)
                            # if st.button("Send Whatsapp Message"):
                            #     no=st.text_input("Enter Phone Number")
                            #     if st.button("Send Message"):
                            #         if no:
                            #             with st.spinner("Sending message..."):
                            #                 try:
                            #                     # Assuming a function send_whatsapp_message exists
                            #                     send_whats_message(no, result)
                            #                     st.success(f"Message sent to {no} successfully!")
                            #                 except Exception as e:
                            #                     st.error(f"Failed to send message: {e}")
                            #         else:
                            #             st.warning("Please enter a valid phone number.")


            # Cleanup temporary images
            cleanup([path for _, path in visualizations])

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        traceback.print_exc() # This will print the full error to the console for debugging
else:
    st.info("Please upload a CSV file to begin your analysis.")