import json
from pathlib import Path

def get_code_generation_prompt(
    data_context,
    user_query,
    result_type,
    needs_multiple_sheets,
    actual_columns,
    numeric_columns,
    categorical_columns,
    worksheet_info,
    worksheet_context,
    active_worksheet,
    prompt_json_path="prompts.all_in_one_with_preamble.json"
):
    """
    Build a full code-generation prompt by dumping the entire JSON prompt file
    plus runtime metadata. No filtering/selection — everything is included.
    Returns a JSON string.
    """
    prompt_file = Path(prompt_json_path)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_json_path}")

    with prompt_file.open("r", encoding="utf-8") as f:
        prompts = json.load(f)

    #runtime_context = {
    #    "user_query": user_query,
    #    "result_type": result_type,
    #    "needs_multiple_sheets": needs_multiple_sheets,
    #    "actual_columns": actual_columns,
    #    "numeric_columns": numeric_columns,
    #    "categorical_columns": categorical_columns,
    #    "worksheet_info": worksheet_info,
    #    "worksheet_context": worksheet_context,
    #    "active_worksheet": active_worksheet,
    #    "data_context": data_context,
    #}

    #full_prompt = {
    #    "prompt_library": prompts,
    #    "runtime_context": runtime_context
    #}

    #print("full prompt loaded successfully")
    #return json.dumps(full_prompt, indent=2)

    return f"""
You are an advanced Data Science assistant proficient in Python Programming, Statistics, Machine Learning & Business Analysis.
You have deep domain knowledge of various business functions like finance, customer operations, marketing, supply chain, procurement, IT and strategy.
Your job is to generate robust python code that answers user's questions in the most helpful manner. This is typically a two step process:
     
Step 1. Analyze the user's query together with the data context to determine the most appropriate analytical approach.
        As part of this enhanced query analysis, determine:
            a. Business function or domain to which the dataset belongs. Some datasets may not be business related so handle them generically.
            b. Analysis type (Descriptive, Diagnostic, Predictive, Prescriptive)
            c. Output format i.e., how best to present the results from code execution so as to be easily understood by the user
            d. Temporal analysis requirements (time series analysis or forecasting)
            e. Statistical & ML complexity involved (basic or advanced stats, regression, classification, clustering etc.)
            f. Visualization requirements i.e., the optimal plot/chart type to use to convey the information 
    
Step 2. Use the analysis conducted in step 1 to generate required code. 

# Code generation guidelines:
    
    {json.dumps(prompts, indent=2)}

# Data Context:

    {data_context}

# User Query:
    
    {user_query}
"""