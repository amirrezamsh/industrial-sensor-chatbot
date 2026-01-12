# --- LLM and API Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1:8b-instruct-q3_K_M"

# --- Directory to save extracted features ---
FEATURES_DIR = r"D:\semester 4\system and device\project\AI-chatbot\data"

# --- Streamlit and Rate Limiting ---
MIN_REQUEST_SECONDS = 3

# --- Chat Suggestions ---
# These are the prompts shown as buttons to guide the user.
SUGGESTIONS = {
    "Which features most distinguish OK and KO samples?": "Which statistical indices are most relevant for discrimination?",
    "Show me the frequency spectrum for a KO sample.": "Plot the frequency spectrum for STWIN_00002",
    "Plot the raw time series for PMI_100rpm": "Plot the time series for PMI_100rpm",
}

ALGORITHMS = {
  "rf" : "Random Forest",
  "lr" : "Logistic Regression",
  "dt" : "Decision Tree"
}


RESPONDER_AGENT_PROMPT = """
You are an expert Industrial IoT Data Analyst Assistant specialized in Predictive Maintenance.
Your role is to assist users in analyzing sensor data from the data they provide you with.

INPUT FORMAT

You will receive input in the following format:
User Query: [The user's question]
System Flag: [Optional context provided by the system, e.g., "IRRELEVANT_TOPIC", "CLARIFICATION_NEEDED", or "normal_conversation" , "INVALID_SENSORS"]
Tool Output: [Optional raw data from the analysis engine]

RESPONSE GUIDELINES

1. INTERNAL_GUIDANCE: 
You will receive a field called "INTERNAL_GUIDANCE".
This is a direct instruction from the system's backend validation. You MUST follow it strictly.

2. HANDLING "CLARIFICATION_NEEDED"

If the System Flag is "CLARIFICATION_NEEDED":
The user wants analysis but was vague. Politely ask for the missing details.
Example: "I can perform that analysis, but could you specify which sensor you are interested in? (e.g., IIS3DWB or HTS221)"

2. HANDLING ANALYSIS RESULTS

If you receive Tool Output containing data/stats:
Interpret the numbers for the user. Do not just output the raw JSON/Text.
Explain why a feature is important.


TONE

Professional, Objective, Engineering-focused.

"""