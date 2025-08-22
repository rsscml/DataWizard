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

    runtime_context = {
        "user_query": user_query,
        "result_type": result_type,
        "needs_multiple_sheets": needs_multiple_sheets,
        "actual_columns": actual_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "worksheet_info": worksheet_info,
        "worksheet_context": worksheet_context,
        "active_worksheet": active_worksheet,
        "data_context": data_context,
    }

    full_prompt = {
        "prompt_library": prompts,
        "runtime_context": runtime_context
    }
    print("full prompt loaded successfully")
    return json.dumps(full_prompt, indent=2)
