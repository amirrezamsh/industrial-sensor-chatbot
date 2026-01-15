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
