
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re  # For preprocessing column names

# --- NLTK Setup (Simplified for Colab) ---
required_resources = ['punkt', 'wordnet', 'omw-1.4']

for resource in required_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        print(f"Downloading {resource}...")
        nltk.download(resource, quiet=True)
        
import pandas as pd
import google.generativeai as genai
import json
import re
import numpy as np
import os
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px # For interactive plots
import plotly.graph_objects as go
from datetime import datetime
import traceback
import io
import re
from typing import List, Dict, Any, Tuple, Optional
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib
matplotlib.use('Agg') # Important for Gradio to prevent GUI conflicts

# SYSTEM_INSTRUCTION_POINTS_BASED_V2 (from your previous version)
LEMMA = WordNetLemmatizer()

# --- Constants ---
ID_KEYWORDS_LEMMATIZED: List[str] = [
    'id', 'phone', 'code', 'zip', 'key', 'account', 'serial',
    'identifier', 'pin', 'ssn', 'uuid', 'guid','mobile','mail','email','address','postal','telephone'
]

MAX_SAMPLE_ROWS_FOR_LLM: int = 3
TOP_N_CATEGORICAL_STATS: int = 5
GEMINI_MODEL_NAME: str = 'gemini-2.0-flash-001'

SYSTEM_INSTRUCTION_POINTS_BASED: str = """
You are an AI Lead Qualification Strategist. Your mission is to meticulously analyze provided business context and data summaries to devise a robust, points-based scoring system for lead qualification. Your output will directly drive an automated process, so precision and adherence to the specified JSON format are paramount.
**Core Task:**
Based on the inputs described below, generate a JSON object containing:
1.  `"scoring_rules"`: A list of specific conditions and the points awarded if a lead meets that condition.
2.  `"thresholds"`: The total point scores that define "High", "Mid", and "Low" lead potential.
**Inputs You Will Receive:**
1.  `product_description`: (string) Detailed information about the product or service being sold.
2.  `product_price_or_fee`: (string, optional) Pricing details.
3.  `ideal_customer_profile`: (string) A description or list of characteristics defining the perfect customer. This is a key document for your analysis.
4.  `csv_headers`: (list of strings) The exact column names from the user's lead data file.
5.  `csv_sample_rows`: (list of lists of strings) The first 1-3 rows of the lead data. This is **only for a general sense of data values and types**; do not base statistical assumptions solely on this sample.
6.  `column_summary_statistics`: (JSON object) This is CRITICAL. It provides aggregated statistics for each column from the *entire dataset*.
    *   **For `dtype: 'numeric'` (Quantitative Numerical Columns):**
        *   Expect fields like `count`, `mean`, `std`, `min`, `25%`, `50%` (median), `75%`, `max`.
        *   Use these to understand the distribution and set meaningful numerical ranges for your scoring rules (e.g., for `greater_than`, `less_than`, `range` conditions).
    *   **For `dtype: 'categorical/object'` (Categorical, Text, or ID-like Columns):**
        *   This type is used for true text/categorical data AND for columns that might be numerically represented in the CSV (like ZIP codes, Phone Numbers, specific IDs) but are best treated as categories for scoring.
        *   Expect fields:
            *   `total_non_na_rows`: Count of non-empty entries.
            *   `distinct_value_count_non_na`: Number of unique values among non-empty entries.
            *   `duplicate_details_non_na`: An object detailing value repetition:
                *   `has_duplicates`: (boolean) True if any value appears more than once.
                *   `items_with_duplicates_and_counts`: (object) `{ "value1": count1, "value2": count2, ... }` for values appearing multiple times.
                *   `count_of_distinct_items_appearing_once`: Number of unique values that appear only once.
            *   `top_n_most_frequent_items`: (object) `{ "valueA": freqA, "valueB": freqB, ... }` for the most common values.
        *   Use these stats to identify significant categories, common identifiers, or patterns of uniqueness/duplication for your scoring rules (e.g., for `equals`, `contains_any_of` conditions).
        * Eg: if in a column there more than one type like => employed , self employed , unemployed so mark them logically employed and self employed +1 and unemployed -1 or -2
          Note this was just an example to tell you but do consider all the types whether distinct or duplicates logically to give them -ve or +ve points according to the situation.
**Your Output: A Single, Valid JSON Object**
The entire response you generate MUST be a single JSON object. Do NOT include any introductory text, explanations, apologies, or markdown formatting (like ` ```json ` or ` ``` `) before or after the JSON structure.
**JSON Structure Details AND Example Output:**
```json
{
  "scoring_rules": [
    {
      "column_name": "Annual_Income_USD",
      "data_type": "numeric",
      "condition_type": "greater_than",
      "values": [100000],
      "points": 2
    },
    {
      "column_name": "Annual_Income_USD",
      "data_type": "numeric",
      "condition_type": "range",
      "values": [60000, 100000],
      "points": 1
    },
    {
      "column_name": "Industry",
      "data_type": "string",
      "condition_type": "equals",
      "values": ["Technology"],
      "points": 1
    },
    {
      "column_name": "Job_Title",
      "data_type": "category_list",
      "condition_type": "contains_any_of",
      "values": ["Manager", "Director", "VP"],
      "points": 1
    },
    {
      "column_name": "Expressed_Interest_Product_X",
      "data_type": "boolean",
      "condition_type": "is_true",
      "values": [],
      "points": 2
    },
    {
      "column_name": "Years_Since_Last_Purchase",
      "data_type": "numeric",
      "condition_type": "greater_than",
      "values": [3],
      "points": -1
    }
  ],
  "thresholds": {
    "high_min_points": 5, // Integer: Score >= this is High
    "mid_min_points": 2  // Integer: Score >= this (and < high_min_points) is Mid. Score < this is Low.
  }
}
"""

# --- Helper Functions ---
def preprocess_column_name(column_name_str: str) -> List[str]:
    """Prepares a column name for lemmatization by splitting and lowercasing."""
    s1 = re.sub(r"([A-Z][a-z]+)", r" \1", column_name_str).strip()
    s2 = re.sub(r"([A-Z]+)", r" \1", s1).strip()
    words = re.sub(r"[_-]+", " ", s2).split()
    return [word.lower() for word in words if word]

def is_identifier_by_name(column_name_str: str, id_keywords: List[str]) -> bool:
    """Checks if column name suggests an identifier using lemmatized ID keywords."""
    processed_words = preprocess_column_name(column_name_str)
    if not processed_words:
        return False
    lemmatized_column_words = {LEMMA.lemmatize(p) for p in processed_words}
    return any(id_keyword in lemmatized_column_words for id_keyword in id_keywords)

# -- Core logic --
def get_column_summary_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for col_name in df.columns:
        col_series = df[col_name]
        col_series_non_na = col_series.dropna()

        treat_as_categorical = False
        is_pandas_numeric = pd.api.types.is_numeric_dtype(col_series)

        if is_pandas_numeric:
            if is_identifier_by_name(col_name, ID_KEYWORDS_LEMMATIZED): # Simplified call
                treat_as_categorical = True
        else:
            treat_as_categorical = True

        if treat_as_categorical:
            current_col_for_cat_stats = col_series_non_na.astype(str) # Ensure string for consistent cat processing
            total_non_na = int(current_col_for_cat_stats.count())
            distinct_non_na = int(current_col_for_cat_stats.nunique())
            val_counts_non_na = current_col_for_cat_stats.value_counts()
            items_with_dupes = {k: int(v) for k, v in val_counts_non_na[val_counts_non_na > 1].items()}
            singly_occurring_count = int(sum(1 for count in val_counts_non_na if count == 1))
            stats[col_name] = {
                'dtype': 'categorical/object',
                'total_non_na_rows': total_non_na,
                'distinct_value_count_non_na': distinct_non_na,
                'duplicate_details_non_na': {
                    'has_duplicates': bool(items_with_dupes),
                    'items_with_duplicates_and_counts': items_with_dupes,
                    'count_of_distinct_items_appearing_once': singly_occurring_count,
                },
                'top_n_most_frequent_items': {
                    k: int(v) for k, v in val_counts_non_na.nlargest(TOP_N_CATEGORICAL_STATS).items()
                },
            }
        else: # True numeric (quantitative)
            if col_series_non_na.empty:
                desc = {'count': 0,'mean': None,'std': None,'min': None,'25%': None,'50%': None,'75%': None,'max': None}
            else:
                desc = col_series_non_na.describe().to_dict()
            stats[col_name] = {
                'dtype': 'numeric',
                'count': int(desc.get('count', 0)),
                'mean': round(desc['mean'], 2) if pd.notna(desc.get('mean')) else None,
                'std': round(desc['std'], 2) if pd.notna(desc.get('std')) else None,
                'min': round(desc['min'], 2) if pd.notna(desc.get('min')) else None,
                '25%': round(desc['25%'], 2) if pd.notna(desc.get('25%')) else None,
                '50%': round(desc['50%'], 2) if pd.notna(desc.get('50%')) else None,
                '75%': round(desc['75%'], 2) if pd.notna(desc.get('75%')) else None,
                'max': round(desc['max'], 2) if pd.notna(desc.get('max')) else None,
            }
    return stats

def prepare_inputs_for_llm(csv_file_path_or_obj, max_sample_rows=3, top_n_categorical_stats=5):
    try:
        # Gradio file object has a .name attribute for the temp path
        df_full = pd.read_csv(csv_file_path_or_obj)
        if df_full.empty:
            try: headers = pd.read_csv(csv_file_path_or_obj, nrows=0).columns.tolist(); return headers, [], {}, df_full
            except Exception: return None, None, None, None # Critical error
        column_stats = get_column_summary_statistics(df_full)
        df_sample = df_full.head(max_sample_rows)
        headers = df_sample.columns.tolist()
        sample_rows = df_sample.values.tolist()
        sample_rows_str = [[str(val) for val in row] for row in sample_rows]
        return headers, sample_rows_str, column_stats, df_full
    except Exception as e:
        print(f"Error in prepare_inputs_for_llm: {e}") # Log for server
        return None, None, None, None # Propagate error

def call_gemini_api(system_instruction: str, product_description: str, product_price: str,
                    ideal_customer_profile: str, csv_headers: List[str], csv_sample_rows: List[List[str]],
                    column_stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Ensure API key is configured (done globally for Gradio app)
    model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME, system_instruction=system_instruction)
    stats_json_str = json.dumps(column_stats, indent=2, default=str)
    prompt_parts = [
        f"Product Description: \"{product_description}\"",
        f"Product Price: \"{product_price if product_price else 'Not provided'}\"",
        f"Ideal Customer Profile:\n{ideal_customer_profile}",
        f"CSV Headers: {csv_headers}",
        f"CSV Sample Rows (first {len(csv_sample_rows)} for preview):\n{csv_sample_rows}",
        f"Column Summary Statistics (from entire dataset):\n{stats_json_str}",
        "\nBased on ALL the information above, provide the scoring rules and thresholds in the specified JSON format. The entire response MUST be only the JSON object."
    ]
    try:
        response = model.generate_content(prompt_parts)
        match = re.search(r'```json\s*([\s\S]+?)\s*```|({[\s\S]+})|(\[[\s\S]+\])', response.text, re.DOTALL)
        if match: json_str = match.group(1) or match.group(2) or match.group(3); return json.loads(json_str) if json_str else None
        try: return json.loads(response.text) # Fallback
        except json.JSONDecodeError: print(f"JSON Decode Error. LLM Raw: {response.text[:500]}..."); return None
    except Exception as e: print(f"Gemini API Error: {e}"); return None

def apply_scoring_rules_to_dataframe(df: pd.DataFrame, llm_rules_json: Dict[str, Any]) -> pd.DataFrame:
    df_scored = df.copy()
    if not llm_rules_json or \
       not isinstance(llm_rules_json.get("scoring_rules"), list) or \
       not isinstance(llm_rules_json.get("thresholds"), dict):
        print("Error: LLM output is missing or has malformed 'scoring_rules' or 'thresholds'.")
        df_scored["lead_score"] = 0
        df_scored["lead_potential"] = "Error_InvalidLLMRules"
        return df_scored

    scoring_rules = llm_rules_json["scoring_rules"]
    thresholds = llm_rules_json["thresholds"]
    df_scored["lead_score"] = 0

    for rule in scoring_rules:
        if not isinstance(rule, dict): print(f"Warning: Skipping invalid rule (not a dictionary): {rule}"); continue
        col_name, rule_dtype, cond_type, cond_values, points = (
            rule.get("column_name"), rule.get("data_type", "string"), rule.get("condition_type"),
            rule.get("values"), rule.get("points")
        )
        if not all([col_name, rule_dtype, cond_type, cond_values is not None, isinstance(points, int)]):
            print(f"Warning: Skipping rule with missing/invalid fields for column '{col_name}': {rule}"); continue
        if col_name not in df_scored.columns:
            print(f"Warning: Column '{col_name}' from rule not in DataFrame. Skipping rule."); continue

        try:
            col_to_compare = pd.to_numeric(df_scored[col_name], errors='coerce') if rule_dtype == "numeric" else df_scored[col_name].astype(str).fillna('')
        except Exception as e_prep: print(f"Warning: Error preparing column '{col_name}' for rule (type: {rule_dtype}). Error: {e_prep}. Skipping."); continue

        conditions_met = pd.Series([False] * len(df_scored), index=df_scored.index)
        try:
            if cond_type == "greater_than":
                val = pd.to_numeric(cond_values[0], errors='coerce')
                if pd.notna(val) and rule_dtype == "numeric": conditions_met = col_to_compare > val
            elif cond_type == "less_than":
                val = pd.to_numeric(cond_values[0], errors='coerce')
                if pd.notna(val) and rule_dtype == "numeric": conditions_met = col_to_compare < val
            elif cond_type == "equals":
                if rule_dtype == "numeric":
                    val = pd.to_numeric(cond_values[0], errors='coerce')
                    if pd.notna(val): conditions_met = col_to_compare == val
                else: conditions_met = col_to_compare.str.lower() == str(cond_values[0]).lower()
            elif cond_type == "range":
                if len(cond_values) == 2 and rule_dtype == "numeric":
                    low = pd.to_numeric(cond_values[0],errors='coerce') if cond_values[0] is not None else -np.inf
                    high = pd.to_numeric(cond_values[1],errors='coerce') if cond_values[1] is not None else np.inf
                    conditions_met = col_to_compare.between(low, high, inclusive="both")
            elif cond_type == "contains_any_of" and rule_dtype != "numeric":
                pattern = '|'.join(re.escape(str(v).lower()) for v in cond_values if v)
                if pattern: conditions_met = col_to_compare.str.lower().str.contains(pattern, na=False)
            elif cond_type == "contains_all_of" and rule_dtype != "numeric":
                match_all = pd.Series([True] * len(df_scored), index=df_scored.index)
                for v_item in cond_values:
                    if v_item: match_all &= col_to_compare.str.lower().str.contains(re.escape(str(v_item).lower()), na=False)
                    else: match_all = pd.Series([False] * len(df_scored), index=df_scored.index); break
                conditions_met = match_all
            elif cond_type == "is_true": conditions_met = col_to_compare.str.lower().isin(['true', '1', 'yes', 't'])
            elif cond_type == "is_false": conditions_met = col_to_compare.str.lower().isin(['false', '0', 'no', 'f', '']) | df_scored[col_name].isnull()
            df_scored.loc[conditions_met & pd.notna(conditions_met), "lead_score"] += points
        except Exception as e_apply: print(f"Error applying rule detail for col '{col_name}', type '{cond_type}': {e_apply}. Points not applied."); continue

    high_min, mid_min = thresholds.get("high_min_points"), thresholds.get("mid_min_points")
    if not (isinstance(high_min, int) and isinstance(mid_min, int)):
        print("Error: Thresholds 'high_min_points' or 'mid_min_points' are missing or not integers.")
        df_scored["lead_potential"] = "Error_InvalidThresholds"; return df_scored
    if mid_min >= high_min: print(f"Warning: mid_min_points ({mid_min}) is not less than high_min_points ({high_min}).")
    df_scored["lead_potential"] = "Low"
    df_scored.loc[df_scored["lead_score"] >= mid_min, "lead_potential"] = "Mid"
    df_scored.loc[df_scored["lead_score"] >= high_min, "lead_potential"] = "High"
    return df_scored
# --- End of core logic functions ---

# --- Rate Limiting Setup ---
import json
from datetime import datetime, date
import ipaddress
from pathlib import Path
import os

# Modify usage file path to work in both local and HF Spaces environments
USAGE_FILE = Path("/tmp/usage.json" if os.getenv("SPACE_ID") else "h:/My Drive/Gromo/Personal sales agent/Final agent/local.json")
MAX_DAILY_USES = 7

def get_current_usage():
    """Load usage data from JSON file, resetting if it's a new day"""
    try:
        if USAGE_FILE.exists():
            with open(USAGE_FILE, 'r') as f:
                data = json.load(f)
                # Reset if stored date != today
                if data.get('date') != str(date.today()):
                    data = {'date': str(date.today()), 'usage': {}}
        else:
            # Ensure directory exists in Spaces
            USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {'date': str(date.today()), 'usage': {}}
            # Initialize file
            with open(USAGE_FILE, 'w') as f:
                json.dump(data, f)
        return data
    except Exception as e:
        print(f"Error reading usage file: {e}")
        return {'date': str(date.today()), 'usage': {}}

def update_usage(ip_address: str) -> Tuple[bool, str]:
    """Update usage for an IP address, return (allowed, message)"""
    try:
        data = get_current_usage()
        
        # Validate IP address
        try:
            ipaddress.ip_address(ip_address)
        except ValueError:
            return False, "Invalid IP address"
        
        # Update usage
        current_usage = data['usage'].get(ip_address, 0)
        if current_usage >= MAX_DAILY_USES:
            return False, f"Daily limit of {MAX_DAILY_USES} uses reached. Please try again tomorrow."
        
        data['usage'][ip_address] = current_usage + 1
        
        # Save updated data
        with open(USAGE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        uses_left = MAX_DAILY_USES - (current_usage + 1)
        return True, f"You have {uses_left} uses remaining today."
    
    except Exception as e:
        print(f"Error updating usage: {e}")
        return True, "Usage tracking error, but request allowed"

# Modify the process_and_qualify_leads_gradio function
def process_and_qualify_leads_gradio(csv_file_obj, product_description, product_price, ideal_customer_profile, progress=gr.Progress(track_tqdm=True), request: gr.Request=None):
    """Added request parameter to get IP address"""
    
    # Check rate limit
    if request:
        client_ip = request.client.host
        allowed, message = update_usage(client_ip)
        if not allowed:
            return None, None, None, None, None, message, None
    
    # Original function continues here
    if not csv_file_obj:
        return None, None, None, None, None, "Error: Please upload a CSV file.", None

    csv_file_path = csv_file_obj.name
    progress(0.1, desc="Preparing inputs...")
    csv_headers, csv_sample_rows, column_stats, df_full = prepare_inputs_for_llm(csv_file_path)

    if df_full is None or csv_headers is None:
        return None, None, None, None, None, "Error: Failed to read/process CSV.", None
    if df_full.empty and not csv_headers:
         return None, None, None, None, None, "Error: CSV file is empty.", None

    status_updates = ["CSV processed. Sending to AI..."]
    progress(0.3, desc=status_updates[-1])
    llm_output_json = call_gemini_api(SYSTEM_INSTRUCTION_POINTS_BASED, product_description, product_price, ideal_customer_profile, csv_headers, csv_sample_rows, column_stats)

    if not llm_output_json or "scoring_rules" not in llm_output_json or "thresholds" not in llm_output_json:
        status_updates.append("Error: AI could not generate valid scoring rules.")
        progress(1.0, desc=status_updates[-1])
        return None, None, None, None, None, "\n".join(status_updates), llm_output_json

    status_updates.append("AI analysis complete. Applying rules...")
    progress(0.7, desc=status_updates[-1])
    df_qualified = apply_scoring_rules_to_dataframe(df_full.copy(), llm_output_json)

    # Generate Plotly plot
    plotly_fig = None
    if "lead_potential" in df_qualified.columns and not df_qualified.empty: # Check if df is not empty
        try:
            lead_data_for_plot = df_qualified.copy()

            # Ensure 'lead_score' is numeric for potential hover data or color scaling
            if 'lead_score' in lead_data_for_plot.columns:
                lead_data_for_plot['lead_score'] = pd.to_numeric(lead_data_for_plot['lead_score'], errors='coerce').fillna(0)

            potential_counts = lead_data_for_plot["lead_potential"].value_counts().reindex(["Low", "Mid", "High"], fill_value=0)
            
            # Calculate average scores for hover data (optional)
            avg_scores = lead_data_for_plot.groupby("lead_potential")['lead_score'].mean().reindex(["Low", "Mid", "High"], fill_value=0).round(1)

            colors = {'Low': '#FF6B6B', 'Mid': '#FFD166', 'High': '#06D6A0'}
            bar_colors = [colors.get(pot, '#CCCCCC') for pot in potential_counts.index] # Use get for safety

            # Custom hover text
            custom_hover_texts = [
                f"<b>Potential:</b> {pot}<br>"
                f"<b>Count:</b> {potential_counts[pot]}<br>"
                f"<b>Avg. Score:</b> {avg_scores[pot]}"
                for pot in potential_counts.index
            ]

            plotly_fig = go.Figure()

            plotly_fig.add_trace(go.Bar(
                x=potential_counts.index,
                y=potential_counts.values,
                text=[f"{val} leads" for val in potential_counts.values], # Text on bars
                textposition='outside', # Place text outside the bar for clarity
                marker_color=bar_colors,
                marker_line_width=1.5,
                marker_line_color='rgba(44,62,80,0.7)', # Darker border for bars
                opacity=0.9,
                hovertext=custom_hover_texts,
                hoverinfo="text", # Use only the custom hover text
                name="Lead Distribution"
            ))

            # Add a title and style the layout
            plotly_fig.update_layout(
                title_text='<b>Lead Potential Distribution</b> ✨',
                title_x=0.5, # Center title
                title_font_size=20,
                xaxis_title='<b>Lead Potential Category</b>',
                yaxis_title='<b>Number of Leads</b>',
                plot_bgcolor='rgba(245, 245, 245, 1)', # Light gray plot background
                paper_bgcolor='rgba(255, 255, 255, 1)', # White paper background
                font=dict(family="Inter, sans-serif", size=13, color="#2c3e50"),
                bargap=0.3, # Gap between bars
                height=450, # Adjust height
                margin=dict(l=60, r=40, t=80, b=60), # Add some margins
                legend_title_text='Metrics',
                xaxis=dict(
                    showgrid=False, # Hide x-axis grid lines
                    categoryorder='array', # Ensure order is Low, Mid, High
                    categoryarray=["Low", "Mid", "High"]
                ),
                yaxis=dict(
                    gridcolor='rgba(220, 220, 220, 0.7)', # Lighter y-axis grid lines
                    zeroline=False
                ),
                hoverlabel=dict( # Style the hover box
                    bgcolor="white",
                    font_size=12,
                    font_family="Inter, sans-serif",
                    bordercolor="rgba(44,62,80,0.3)"
                )
            )
            # Add an annotation for total leads
            total_leads = potential_counts.sum()
            plotly_fig.add_annotation(
                text=f"Total Leads Analyzed: <b>{total_leads}</b>",
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.02,
                y=1.08, # Position above the title
                font=dict(size=12, color="#555")
            )

        except Exception as e_plot:
            status_updates.append(f"Warning: Plotly plot error - {e_plot}")
            print(f"Plotting Error: {e_plot}") # For server-side debugging
            plotly_fig = None
    elif df_qualified.empty:
        status_updates.append("Info: No data to plot (qualified dataframe is empty).")
        plotly_fig = go.Figure() # Empty figure
        plotly_fig.update_layout(title_text='No Data to Display', title_x=0.5, xaxis_visible=False, yaxis_visible=False,
                                 plot_bgcolor='rgba(245, 245, 245, 1)', paper_bgcolor='rgba(255, 255, 255, 1)')
        
    # --Prepare downloadable files--
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files = {}

    def save_df_to_path(df_to_save, base_name):
        filename = f"{base_name}_{timestamp}.csv"
        df_to_save.to_csv(filename, index=False)
        return filename # Return the path for Gradio

    output_files['all'] = save_df_to_path(df_qualified, "all_qualified_leads")
    if "lead_potential" in df_qualified.columns:
        output_files['high'] = save_df_to_path(df_qualified[df_qualified["lead_potential"] == "High"], "high_potential_leads")
        output_files['mid'] = save_df_to_path(df_qualified[df_qualified["lead_potential"] == "Mid"], "mid_potential_leads")
        output_files['low'] = save_df_to_path(df_qualified[df_qualified["lead_potential"] == "Low"], "low_potential_leads")
    else: # Handle case where lead_potential column might not exist due to errors
        empty_df = pd.DataFrame()
        output_files['high'] = save_df_to_path(empty_df, "high_potential_leads_empty")
        output_files['mid'] = save_df_to_path(empty_df, "mid_potential_leads_empty")
        output_files['low'] = save_df_to_path(empty_df, "low_potential_leads_empty")


    status_updates.append("Processing complete! Files ready for download.")
    progress(1.0, desc=status_updates[-1])

    return (
        output_files.get('all'),
        output_files.get('high'),
        output_files.get('mid'),
        output_files.get('low'),
        plotly_fig, # Plotly figure object
        "\n".join(status_updates),
        llm_output_json
    )

# --- Configure API Key ---
API_KEY_CONFIGURED = False
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        API_KEY_CONFIGURED = True
        print("Gemini API key loaded from .env file.")
    else:
        print("WARNING: GOOGLE_API_KEY not found in .env file.")
except ImportError:
    print("WARNING: python-dotenv not installed. Please install with: pip install python-dotenv")
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # Fallback to system environment variable
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        API_KEY_CONFIGURED = True
        print("Gemini API key loaded from system environment.")

if not API_KEY_CONFIGURED:
    print("WARNING: GOOGLE_API_KEY not found. AI analysis will fail.")

# --- Gradio UI with Blocks ---

css = """
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    background-color: #f4f7f6; /* Light gray-ish background for the page */
    color: #333;
    line-height: 1.6;
}
.gradio-container {
    max-width: 1100px !important; /* Wider for desktop */
    margin: 2rem auto !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08) !important; /* Softer, more spread shadow */
    background-color: #ffffff !important; /* White background for the main app container */
    padding: 1.5rem !important; /* Add some padding inside the container */
}
/* --- Section Titles --- */
.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #2c3e50; /* Darker, more professional blue-gray */
    text-align: center;
    margin-bottom: 25px;
    padding-bottom: 12px;
    border-bottom: 3px solid #3498db; /* Primary theme color */
}
.subtitle-markdown {
    text-align: center;
    margin-bottom: 25px;
    font-size: 1.1em;
    color: #555;
}
.step-title { /* For STEP 1, STEP 2 */
    font-size: 1.5em;
    font-weight: 600;
    color: #3498db; /* Primary theme color */
    margin-bottom: 15px;
    /* border-left: 4px solid #3498db;
    padding-left: 10px; */
}
.output-section-title { /* For titles above plot/downloads */
    font-size: 1.3em;
    font-weight: 600;
    color: #2c3e50;
    margin-top: 25px;
    margin-bottom: 12px;
}
/* --- Input Components & Panels --- */
.gr-input input, .gr-input textarea, .gr-file input[type="file"], .gr-dropdown select {
    border-radius: 10px !important;
    border: 1px solid #ddd !important;
    padding: 12px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.03) inset;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.gr-input input:focus, .gr-input textarea:focus, .gr-dropdown select:focus {
    border-color: #3498db !important; /* Primary theme color */
    box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2) !important; /* Focus ring */
}
.gr-panel, .gradio-accordion { /* For panels and accordions */
    border-radius: 15px !important;
    padding: 25px !important;
    background-color: #fdfdfd !important; /* Slightly off-white for panels */
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05) !important;
    border: 1px solid #e0e0e0;
    margin-top: 20px;
}
/* --- Buttons --- */
.gr-button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
    transition: all 0.3s ease !important;
    border: none !important; /* Remove default border for custom shadow approach */
    box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08) !important;
}
.gr-button.gr-button-primary { /* Primary submit button */
    background: linear-gradient(135deg, #5c67f2 0%, #3498db 100%) !important; /* Violet-Blue gradient */
    color: white !important;
}
.gr-button.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08) !important;
}
.gr-button.gr-button-secondary { /* For download buttons */
    background-color: #ffffff !important;
    color: #3498db !important; /* Primary theme color for text */
    border: 2px solid #3498db !important; /* Primary theme color for border */
}
.gr-button.gr-button-secondary:hover {
    background-color: #e9f5ff !important; /* Very light blue on hover */
    transform: translateY(-1px) !important;
}
/* --- Specific UI Elements --- */
#logo-image img { /* Assuming the image component renders an img tag */
    border-radius: 12px;
    object-fit: cover;
}
.status-box {
    background-color: #e9f5ff; /* Light blue background */
    border-left: 5px solid #3498db; /* Primary theme color */
    padding: 15px;
    border-radius: 10px;
    margin-top: 20px;
    color: #2c3e50;
}
.json-output-box .json-formatter-value { /* Improve JSON viewer colors */
    color: #c5f6fa !important; /* Light cyan for values */
}
.json-output-box .json-formatter-key {
    color: #ffc078 !important; /* Light orange for keys */
}
.gradio-plot { /* Style the container of the plot */
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    background-color: #fff;
}
/* Responsive adjustments (example for a 2-column layout on wider screens) */
/* This is a simplified example. True responsive layout shifts often need more complex CSS or JS. */
@media (min-width: 768px) {
    .input-section-column { /* A class you might add to columns containing inputs */
        /* Potentially adjust flex properties if inputs were in a gr.Row */
    }
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), css=css, title="AI Lead Analyzer") as demo:
    gr.Markdown("# AI Lead Analyzer", elem_classes="section-title")
    gr.Markdown("Empower your sales strategy by identifying high-potential leads with AI precision.", elem_classes="subtitle-markdown")

    with gr.Row(variant="panel"): # Wrap input section in a panel-like row
        with gr.Column(scale=1, min_width=300): # Logo/Image Column
             gr.Image("Vista ai image.png",
                     # Replace with your actual logo URL or path
                     label=None, show_label=False, height=250, interactive=False, show_download_button=False,
                     elem_id="logo-image")
             gr.Markdown("<h3 style='text-align:center; color:#34495e;'>Vista Lead AI</h3>",)

        with gr.Column(scale=3, min_width=500): # Inputs Column
            gr.Markdown("### STEP 1: Provide Your Data & Product Info", elem_classes="step-title")
            with gr.Row():
                csv_file = gr.File(label="Upload Lead CSV File", file_types=[".csv"], elem_id="csv_upload", scale=2)
                product_price_input = gr.Textbox(label="Product Price/Fee (Optional)", placeholder="e.g., $99/mo", scale=1)

            product_desc_input = gr.Textbox(label="Product Description", lines=4, placeholder="Describe your product/service...")
            icp_input = gr.Textbox(label="Ideal Customer Profile (ICP)", lines=5, placeholder="List characteristics of your ideal customer...")

    submit_button = gr.Button(" Analyze Leads & Generate Insights ", variant="primary", elem_id="submit_btn_main", size="lg")

    gr.Markdown("---", elem_classes="hr-markdown")
    gr.Markdown("### STEP 2: Review Results & Download", elem_classes="step-title")

    with gr.Tabs():
        with gr.TabItem("📊 Lead Distribution Chart"):
            gr.Markdown("#### Lead Potential Visualization", elem_classes="output-section-title")
            output_plotly_plot = gr.Plot(label="Lead Potential Chart") # For Plotly output

        with gr.TabItem("📁 Download Qualified Leads"):
            gr.Markdown("#### Download Files", elem_classes="output-section-title")
            with gr.Row():
                output_file_all = gr.File(label="All Qualified Leads (.csv)", interactive=True, elem_classes="download-button-class") # Add specific class if needed
                output_file_high = gr.File(label="High Potential Leads (.csv)", interactive=True)
            with gr.Row():
                output_file_mid = gr.File(label="Mid Potential Leads (.csv)", interactive=True)
                output_file_low = gr.File(label="Low Potential Leads (.csv)", interactive=True)

        with gr.TabItem("💡 AI Scoring Logic"):
            gr.Markdown("#### Raw AI Scoring Rules (JSON)", elem_classes="output-section-title")
            raw_llm_output_json = gr.JSON(label="LLM Generated Scoring Logic", elem_classes="json-output-box")

    status_md = gr.Markdown("Status: Ready to process your data.", elem_classes="status-box")


    submit_button.click(
        fn=process_and_qualify_leads_gradio,
        inputs=[csv_file, product_desc_input, product_price_input, icp_input],
        outputs=[
            output_file_all, output_file_high, output_file_mid, output_file_low,
            output_plotly_plot,
            status_md, raw_llm_output_json
        ],
        api_name=False  # Disable API endpoint to prevent circumventing rate limit
    )
    demo.queue() # Enable queuing for the whole app

    gr.Markdown("<p style='text-align:center; font-size:0.9em; color:#7f8c8d; margin-top:2rem;'>Powered By Micronova -AI Battalion</p>")

if __name__ == '__main__':
    if not API_KEY_CONFIGURED:
        print("*********************************************************************")
        print("WARNING: Gemini API Key is not configured. The application will run,")
        print("but the core AI functionality will fail. Please set your")
        print("GOOGLE_API_KEY in Colab Secrets or as an environment variable.")
        print("*********************************************************************")
    if os.getenv("SPACE_ID"):
        demo.launch(
            debug=True,
            # max_requests_per_minute=7,  # Built-in rate limiting
            # max_concurrent_requests=7
        )
    else:
        demo.launch(debug=True, share=False)