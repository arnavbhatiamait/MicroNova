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

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Model configurations
GEMINI_MODEL = 'gemini-1.5-flash-latest'
LLAMA_MODEL = 'llava:7b'
GEMINI_AVAILABLE = False

# Configure Google Generative AI
try:
    if api_key := os.environ.get('GOOGLE_API_KEY'):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        GEMINI_AVAILABLE = True
        st.success("Gemini model is available")
except Exception as e:
    st.error(f"Error configuring Gemini: {e}")

# Helper functions
def save_fig(fig):
    f = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fig.savefig(f.name, bbox_inches='tight')
    plt.close(fig)
    return f.name

def df_into_string(df, max_rows=5):
    buf = io.StringIO()
    df.info(buf=buf)
    schema = buf.getvalue()
    head = df.head(max_rows).to_markdown(index=False)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_info = "No missing values" if missing.empty else f"Missing values: {missing.to_dict()}"
    return f"### Schema:\n{schema}\n### Preview:\n{head}\n### {missing_info}"

async def gemini_text_analysis(prompt_type, df_context):
    if not GEMINI_AVAILABLE:
        return "Gemini model is not available"
    prompts = {
        "swot": f"Perform a SWOT analysis based on the following microfinance sales data:\n{df_context}",
        "best_customers": f"List the top 5 best customers based on repayment and loan amount:\n{df_context}",
        "worst_customers": f"List the 5 worst customers based on repayment delays or defaults:\n{df_context}",
        "future": f"Suggest future steps and strategies for a microfinance company based on this data:\n{df_context}"
    }
    try:
        model=genai.GenerativeModel(model_name=GEMINI_MODEL)
        res=await model.generate_content_async(
            prompts.get(prompt_type),
            generation_config=genai.types.GenerationConfig(max_output_tokens=1500, temperature=0.3)
        )
        return res.text if res.parts else "No response from Gemini model"
    except Exception as e:
        return f"Error: {e}"

async def ollama_text_analysis(prompt_type, df_context):
    prompts = {
        "swot": f"Perform a SWOT analysis based on the following microfinance sales data:\n{df_context}",
        "best_customers": f"List the top 5 best customers based on repayment and loan amount:\n{df_context}",
        "worst_customers": f"List the 5 worst customers based on repayment delays or defaults:\n{df_context}",
        "future": f"Suggest future steps and strategies for a microfinance company based on this data:\n{df_context}"
    }
    try:
        res=ollama.generate(prompt=prompts.get(prompt_type), model=LLAMA_MODEL)
        return res.response if res else "No response from Ollama model"
    except Exception as e:
        return f"Error: {e}"

def visual_generation(df):
    visualizations = []
    try:
        # Sales trend over time
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df_time = df.dropna(subset=['date'])
            sales_over_time = df_time.groupby('date')['loan_amount'].sum()
            fig, ax = plt.subplots(figsize=(10,6))
            sales_over_time.plot(ax=ax)
            ax.set_title("Total Loan Amount Over Time")
            path = save_fig(fig)
            visualizations.append(("Loan Amount Over Time", path))
        # Repayment status
        if 'repayment_status' in df.columns:
            status_counts = df['repayment_status'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=status_counts.index, y=status_counts.values, ax=ax)
            ax.set_title("Repayment Status Distribution")
            path = save_fig(fig)
            visualizations.append(("Repayment Status Distribution", path))
        # Loan amount distribution
        if 'loan_amount' in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df['loan_amount'], kde=True, ax=ax)
            ax.set_title("Loan Amount Distribution")
            path = save_fig(fig)
            visualizations.append(("Loan Amount Distribution", path))
        # Top customers
        if 'customer_id' in df.columns and 'loan_amount' in df.columns:
            top_customers = df.groupby('customer_id')['loan_amount'].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots()
            top_customers.plot(kind='bar', ax=ax)
            ax.set_title("Top 10 Customers by Loan Amount")
            path = save_fig(fig)
            visualizations.append(("Top Customers", path))
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, ax=ax)
            ax.set_title("Correlation among Numeric Features")
            path = save_fig(fig)
            visualizations.append(("Numeric Features Correlation", path))
        if 'Order Date' in df.columns:
            df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
            df_time = df.dropna(subset=['Order Date']).copy() # Create a copy to avoid modifying the original DataFrame
            if not df_time.empty: # Check if the dataframe is empty
                sales_over_time = df_time.groupby('Order Date')['Total Revenue'].sum()
                fig, ax = plt.subplots(figsize=(10, 6))
                sales_over_time.plot(ax=ax)
                ax.set_title("Total Revenue Over Time")
                ax.set_xlabel("Date")
                ax.set_ylabel("Total Revenue")
                path = save_fig(fig)
                visualizations.append(("Total Revenue Over Time", path))


        # Sales by Item Type
        if 'Item Type' in df.columns:
            item_sales = df.groupby('Item Type')['Total Revenue'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            item_sales.plot(kind='bar', ax=ax)
            ax.set_title("Total Revenue by Item Type")
            ax.set_xlabel("Item Type")
            ax.set_ylabel("Total Revenue")
            path = save_fig(fig)
            visualizations.append(("Total Revenue by Item Type", path))


        # Sales by Region
        if 'Region' in df.columns:
            region_sales = df.groupby('Region')['Total Revenue'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            region_sales.plot(kind='bar', ax=ax)
            ax.set_title("Total Revenue by Region")
            ax.set_xlabel("Region")
            ax.set_ylabel("Total Revenue")
            path = save_fig(fig)
            visualizations.append(("Total Revenue by Region", path))

        # Order Priority Distribution
        if 'Order Priority' in df.columns:
            priority_counts = df['Order Priority'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=priority_counts.index, y=priority_counts.values, ax=ax)
            ax.set_title("Order Priority Distribution")
            ax.set_xlabel("Order Priority")
            ax.set_ylabel("Count")
            path = save_fig(fig)
            visualizations.append(("Order Priority Distribution", path))

        # Total Profit vs Total Revenue
        if 'Total Profit' in df.columns and 'Total Revenue' in df.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(x='Total Revenue', y='Total Profit', data=df, ax=ax)
            ax.set_title("Total Profit vs Total Revenue")
            ax.set_xlabel("Total Revenue")
            ax.set_ylabel("Total Profit")
            path = save_fig(fig)
            visualizations.append(("Total Profit vs Total Revenue", path))

        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Correlation Matrix of Numeric Features")
            path = save_fig(fig)
            visualizations.append(("Correlation Matrix", path))
    
    except Exception as e:
        st.error(f"Error in visualization: {e}")
    return visualizations

def cleanup(files):
    for file in files:
        try:
            os.remove(file)
        except Exception:
            pass

# Streamlit App
st.title("Microfinance Data Analysis Assistant")

uploaded_file = st.file_uploader("Upload your sales data CSV file", type=["csv"])

if uploaded_file:
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
                info_str = buf.getvalue()
            st.text_area("Data Schema & Info", value=info_str, height=200)

            # Generate visualizations
            st.subheader("Data Visualizations")
            visualizations = []
            try:
                visualizations = visual_generation(df)
                for title, path in visualizations:
                    st.image(path, caption=title)
            except Exception as e:
                st.error(f"Error generating visualizations: {e}")

            # Prepare info string for analysis
            info_str = df_into_string(df)

            # Analysis buttons
            if st.button("Perform SWOT Analysis"):
                if GEMINI_AVAILABLE:
                    swot_result = asyncio.run(gemini_text_analysis("swot", info_str))
                else:
                    swot_result = "Gemini model is not available."
                st.markdown("### SWOT Analysis")
                st.write(swot_result)

            if st.button("List Top Customers"):
                if GEMINI_AVAILABLE:
                    top_customers = asyncio.run(gemini_text_analysis("best_customers", info_str))
                else:
                    top_customers = "Gemini model is not available."
                st.markdown("### Top Customers")
                st.write(top_customers)

            if st.button("List Worst Customers"):
                if GEMINI_AVAILABLE:
                    worst_customers = asyncio.run(gemini_text_analysis("worst_customers", info_str))
                else:
                    worst_customers = "Gemini model is not available."
                st.markdown("### Worst Customers")
                st.write(worst_customers)

            if st.button("Suggest Future Strategies"):
                if GEMINI_AVAILABLE:
                    future_strat = asyncio.run(gemini_text_analysis("future", info_str))
                else:
                    future_strat = "Gemini model is not available."
                st.markdown("### Future Strategies")
                st.write(future_strat)

            # Cleanup temporary images
            cleanup([path for _, path in visualizations])

    except Exception as e:
        st.error(f"An error occurred: {e}")
        traceback.print_exc()
else:
    st.info("Please upload a CSV file to begin.")
