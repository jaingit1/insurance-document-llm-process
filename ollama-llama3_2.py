import json
import fitz  # PyMuPDF for extracting text from PDF 
from langchain_ollama import OllamaLLM
import time  # For tracking response time
import re 

# Function to extract the fleet and mileage from pdf text
def extract_fleet_and_mileage_details(text):
    """
    Extract fleet count, annual mileage, daily mileage, reimbursed mileage, and state from text.

    Args:
        text (str): Input text from a PDF.

    Returns:
        dict: Extracted details.
    """
    # Fleet count
    fleet_pattern = r"(?i)fleet[:\-]?\s*(\d+)\s*total units"
    fleet_match = re.search(fleet_pattern, text)
    fleet_count = int(fleet_match.group(1)) if fleet_match else None

    # Annual mileage
    annual_mileage_pattern = r"(?i)(annual|HNO)\s*mileage[:\-]?\s*([\d,]+)"
    annual_mileage_match = re.search(annual_mileage_pattern, text)
    annual_mileage = int(annual_mileage_match.group(2).replace(",", "")) if annual_mileage_match else None

    # Daily mileage (derived from annual mileage if available)
    daily_mileage = annual_mileage / 250 if annual_mileage else None

    # Reimbursed mileage
    reimbursed_mileage_pattern = r"(?i)reimbursed\s*mileage[:\-]?\s*([\d,]+)"
    reimbursed_mileage_match = re.search(reimbursed_mileage_pattern, text)
    reimbursed_mileage = int(reimbursed_mileage_match.group(1).replace(",", "")) if reimbursed_mileage_match else None

    # State breakout
    state_pattern = r"(?i)state\s*breakout\s*[-–]?\s*(.*)"
    state_match = re.search(state_pattern, text)
    state = state_match.group(1).strip() if state_match else None

    # Split state list if multiple states are mentioned
    states = [s.strip() for s in state.split("&")] if state else []

    # Return extracted details
    return {
        "Total Fleet": fleet_count,
        "Annual Mileage": annual_mileage,
        "Daily Mileage": round(daily_mileage, 2) if daily_mileage else None,
        "Reimbursed Mileage": reimbursed_mileage,
        "States": states
    }

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    pdf_document = fitz.open(file_path)
    pdf_text = ""
    for page in pdf_document:
        pdf_text += page.get_text()
    pdf_document.close()
    return pdf_text

# Process text with DeepSeek model and track response time
def process_pdf_with_llama(pdf_text, model_name="llama3.2"):
    """
    Process extracted PDF text using DeepSeek and track response time.

    Args:
        pdf_text (str): Extracted text from the PDF.
        model_name (str): Name of the DeepSeek model.

    Returns:
        dict: Extracted data points from the text.
        float: Response time in seconds.
    """
    # Initialize Ollama DeepSeek model
    llm = OllamaLLM(model=model_name)
    
    # Define the prompt with consistent formatting and JSON instruction
    prompt = f"""
    Read the following details from the given text and return the output in a JSON format:
    - Total Fleet: Total number of vehicles in the fleet. Return an integer value if present in given text 
    - Annual Mileage: Annual mileage of the fleet, Return an integer value if present in given text otherwise Null
    - Daily Mileage: Annual mileage of the fleet, Return an integer or float value if present in given text otherwise Null
    - Reimbursed Mileage: Reimbursed mileage of the fleet, Return an integer value if present in given text otherwise Null
    - States: Return both the full name and abbreviation for all states as a list of dictionaries (e.g. Full Name: Oregon, Abbreviation: OR)
    Text to process:
    {pdf_text}
    """

    # Track response time
    start_time = time.time()
    response = llm.invoke(prompt)
    end_time = time.time()
    response_time = end_time - start_time  # Calculate response time in seconds
    print(response)
    
    # Parse response dynamically
    extracted_data = {}
    try:
        # Regex pattern to capture JSON structure
        json_pattern = r'{(?:[^{}]|(?:{.*}))*}'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            extracted_data = json.loads(match.group(0))  # Safely parse the JSON part
        else:
            print("No valid JSON found!")

        # # response = response.strip().encode("utf-8").decode("utf-8")
        # if "```json" in response and "```" in response:
        #     start_idx = response.index("```json") + len("```json")
        #     end_idx = response.index("```", start_idx)
        #     json_content = response[start_idx:end_idx].strip()
        #     extracted_data = json.loads(json_content)
        # elif "```" in response:
        #     start_idx = response.index("```") + len("```")
        #     end_idx = response.index("```", start_idx)
        #     json_content = response[start_idx:end_idx].strip()
        #     extracted_data = json.loads(json_content)
        # else:
        #     print("Error JSON block not found in response.")
    except Exception as e:
        print(f"Error parsing response: {e}")
    return extracted_data, response_time

# Function to calculate Per Person Trip (PPT)
def calculate_ppt(total_fleet, annual_mileage=None, daily_mileage=None, reimbursed_mileage=None, states="CO"):
    """
    Calculate Per Person Trip (PPT) based on input data and make a decision to approve or decline.

    Args:
        total_fleet (int): Total number of vehicles in the fleet.
        annual_mileage (float): Annual mileage of the fleet, if provided.
        daily_mileage (float): Daily mileage of the fleet, if provided.
        reimbursed_mileage (float): Reimbursed mileage of the fleet, if provided.
        state (str): State abbreviation where vehicles are operating.

    Returns:
        str: "Approve" or "Decline" decision based on calculations.
    """
    # Determine the reimbursement rate based on the state
    reimbursement_rate = 0.655
    for state_entry in states:
        state_name = state_entry["Full Name"]
        actual_abbreviation = state_entry["Abbreviation"]
        if actual_abbreviation != None and actual_abbreviation.strip().upper() in ["OR", "WA"]:
            reimbursement_rate = 0.8
        elif state_name != None and state_name.strip().upper() in ["OREGON", "WASHINGTON"]:
            reimbursement_rate = 0.8
         

    # Calculate annual mileage if daily mileage is provided
    if daily_mileage:
        annual_mileage = daily_mileage * 250  # Assuming 250 working days

    # Calculate PPT    
    if reimbursed_mileage:
        ppt = reimbursed_mileage / (reimbursement_rate * 15000)
    elif annual_mileage:
        ppt = annual_mileage / 15000
    else:
        raise ValueError("Either annual mileage or reimbursed mileage must be provided.")

    # Make decision
    return {
        "decision": "Approve" if total_fleet > ppt else "Decline",
        "ppt": ppt
        }
def save_results_to_json(predicted_values, output_file="predicted_results.json"):
    """
    Save predicted values from the LLM model into a JSON file.

    Args:
        predicted_values (dict): The predicted data as a dictionary.
        output_file (str): Path to the output JSON file.

    Returns:
        None
    """
    try:
        # Save the predicted values to a JSON file
        with open(output_file, "w") as json_file:
            json.dump(predicted_values, json_file, indent=4)
        print(f"{output_file} saved")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

def read_json_file(file_path):
    """
    Reads a JSON file and returns its contents.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: Parsed JSON data as a Python object.
    """
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return None

def evaluate_results(predicted, ground_truth):
    """
    Evaluates predicted JSON data against ground truth, including accuracy metrics.

    Args:
        predicted (list of dict): Predicted values.
        ground_truth (list of dict): Ground truth values.

    Returns:
        dict: Detailed evaluation results and accuracy metrics.
    """
    total_fields = 0
    matched_fields = 0
    evaluation = []

    for i, (pred, truth) in enumerate(zip(predicted, ground_truth)):
        result = {}
        for key in truth.keys():
            total_fields += 1  # Increment total fields for comparison
            if key == "States":  # Special handling for state comparison
                # Extract abbreviations from predicted states
                predicted_states = [
                    state.get("Abbreviation") for state in pred[key] if isinstance(state, dict) and "Abbreviation" in state
                ]
                ground_truth_states = truth[key]

                # Compare states as sets
                match = set(predicted_states) == set(ground_truth_states)
                if match:
                    matched_fields += 1  # Increment matched fields if states match
                result[key] = {
                    "match": match,
                    "predicted": predicted_states,
                    "ground_truth": ground_truth_states
                }

            elif isinstance(truth[key], (int, float, type(None))):  # Numeric or None fields
                # Handle cases where one or both values are None
                if truth[key] is None or pred[key] is None:
                    match = truth[key] == pred[key]  # Both must be None to match
                else:
                    # Use a tolerance for numeric comparison
                    tolerance = 0.01  # Allow ±1% tolerance
                    match = abs(pred[key] - truth[key]) <= tolerance * truth[key]
                if match:
                    matched_fields += 1  # Increment matched fields if numeric values match
                result[key] = {
                    "match": match,
                    "predicted": pred[key],
                    "ground_truth": truth[key]
                }

            else:  # Generic comparison for other fields
                match = pred[key] == truth[key]
                if match:
                    matched_fields += 1  # Increment matched fields for generic matches
                result[key] = {
                    "match": match,
                    "predicted": pred[key],
                    "ground_truth": truth[key]
                }

        # Add result for the current record
        evaluation.append({
            "Record": i + 1,
            "Evaluation": result
        })

    # Calculate accuracy
    accuracy = (matched_fields / total_fields) * 100 if total_fields > 0 else 0

    return {
        "Evaluation Results": evaluation,
        "Accuracy Metrics": {
            "Total Fields": total_fields,
            "Matched Fields": matched_fields,
            "Accuracy (%)": accuracy
        }
    }



# Main Workflow
if __name__ == "__main__":
    # Example: Path to the PDF file
    pdf_file_path_1 = "./Input_Data/Sample_American_RE_company.pdf"
    pdf_file_path_2 = "./Input_Data/Sample15.pdf"

    # Step 1: Extract text from the PDF
    pdf_text_1 = extract_text_from_pdf(pdf_file_path_1)
    pdf_text_2 = extract_text_from_pdf(pdf_file_path_2)
    # # Step 2: Process the extracted text with regular expressions and extract mileage and fleet information in json format
    extracted_data_pdf_1 = extract_fleet_and_mileage_details(pdf_text_1)
    extracted_data_pdf_2 = extract_fleet_and_mileage_details(pdf_text_2)

    # with llama to extract key data points
    
    
    print("*****Pdf 1 Process*****")
    print("=====Regular Expression Extracted Pdf Data=====")
    print(f"Regular Expression Extracted Data: {extracted_data_pdf_1}")
    llm_extracted_data_pdf_1, response_time_pdf_1 = process_pdf_with_llama(pdf_text_1)
    print("=====LLM Extracted Pdf Data=====")
    print(f"Response time: {response_time_pdf_1:.2f} seconds")
    print(f"LLM Extracted Pdf Data: {llm_extracted_data_pdf_1}")
  
    # Step 3: Use extracted data to calculate PPT and make a decision
    decision_pdf_1 = calculate_ppt(
        total_fleet=llm_extracted_data_pdf_1.get("Total Fleet"),
        annual_mileage=llm_extracted_data_pdf_1.get("Annual Mileage"),
        daily_mileage=llm_extracted_data_pdf_1.get("Daily Mileage"),
        reimbursed_mileage=llm_extracted_data_pdf_1.get("Reimbursed Mileage"),
        states=llm_extracted_data_pdf_1.get("States")

    )

    # # Step 4: Output the decision and response time
    print(f"The Pdf 1 decision is: {decision_pdf_1}")

    
    print("*****Pdf 2 Process*****")
    print("=====Regular Expression Extracted Pdf Data=====")
    print(f"Regular Expression Extracted Data: {extracted_data_pdf_2}")
    llm_extracted_data_pdf_2, response_time_pdf_2 = process_pdf_with_llama(pdf_text_2)
    print("=====LLM Extracted Pdf Data=====")
    print(f"Response time: {response_time_pdf_2:.2f} seconds")
    print(f"LLM Extracted Pdf Data: {llm_extracted_data_pdf_2}")

    # Step 3: Use extracted data to calculate PPT and make a decision
    decision_pdf_2 = calculate_ppt(
        total_fleet=llm_extracted_data_pdf_2.get("Total Fleet"),
        annual_mileage=llm_extracted_data_pdf_2.get("Annual Mileage"),
        daily_mileage=llm_extracted_data_pdf_2.get("Daily Mileage"),
        reimbursed_mileage=llm_extracted_data_pdf_2.get("Reimbursed Mileage"),
        states=llm_extracted_data_pdf_2.get("States")
    )

    # # Step 4: Output the decision and response time
    print(f"The Pdf 2 decision is: {decision_pdf_2}")
    
    print("+++++Compare predicted llm extracted data with Ground Truth+++++")
    predicted_data = [llm_extracted_data_pdf_1, llm_extracted_data_pdf_2]
    # predicted_data_json = json.loads(predicted_data)
    save_results_to_json(predicted_data)
    ground_truth_data = [extracted_data_pdf_1, extracted_data_pdf_2]
    # ground_truth_data_json = json.loads(ground_truth_data)
    save_results_to_json(ground_truth_data, "ground_truth_results.json")

    print("!!!!!Evaluation!!!!!")
    predicted_file = "predicted_results.json"
    ground_truth_file = "ground_truth_results.json"
    predicted_file_data = read_json_file(predicted_file)
    ground_truth_file_data = read_json_file(ground_truth_file)
    # results = evaluate_results(predicted_file_data,ground_truth_file_data)
    # print(json.loads(results))