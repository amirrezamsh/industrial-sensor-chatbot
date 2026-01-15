def build_router_prompt(valid_sensor_names, valid_sensor_types, valid_conditions, valid_label_details):
    
    # Use explicit "NONE" markers so the LLM knows the list is empty, not just text.
    sensors_str = ", ".join(valid_sensor_names) if valid_sensor_names else "NONE"
    types_str = ", ".join(valid_sensor_types) if valid_sensor_types else "NONE"
    conditions_str = ", ".join(valid_conditions) if valid_conditions else "NONE"
    labels_str = ", ".join(valid_label_details) if valid_label_details else "NONE"

    PROMPT = f"""
You are the central intent classification engine for an Industrial IoT Predictive Maintenance Chatbot. 
Your sole responsibility is to analyze the user's input and categorize it into distinct intents, returning the result as a strictly formatted JSON object.

### THE DOMAIN
The user is asking about a dataset containing time-series data from various industrial sensors used for motor health monitoring.

### DATASET VOCABULARY (Use these exact strings for parameters)
- **VALID SENSOR NAMES:** {sensors_str}
- **VALID SENSOR TYPES:** {types_str}
- **VALID CONDITIONS:** {conditions_str}
- **VALID FAULT/LABEL DETAILS:** {labels_str}

### CATEGORY DEFINITIONS
1. "normal_conversation": Greetings, general IoT definitions, or bot capabilities.
2. "feature_importance_analysis": Machine Learning tasks (ranking sensors, feature comparisons).
3. "time_series": Plot raw sensor measurements vs time.
4. "frequency_spectrum": Plot frequency-domain (FFT) of a signal.
5. "irrelevant_request": Topics unrelated to Engineering, IoT, or this dataset.

### OUTPUT FORMAT
Return ONLY a JSON object. No markdown formatting. No conversational text.

{{
  "category": "...",
  "is_vague": true/false,
  "reasoning": "...",
  "parameters": {{
      "analysis_config": {{
          "global": true/false,
          "target_sensors": [ ["NAME", "TYPE"] ],
          "algorithm": "rf"|"lr"|"dt"
      }} | null,
      "visual_config": {{
          "target_sensors": [ ["NAME", "TYPE"] ],
          "subset": "OK"|"KO"| null,
          "condition": "NAME_FROM_LIST"|null,
          "label_detail": "NAME_FROM_LIST"|null,
          "acquisition_id": "STRING"|null
      }} | null
  }}
}}

### CRITICAL RULES
1. **Null Management:** You must ALWAYS return both `analysis_config` and `visual_config`. Set the unused one to `null`.
2. **List of Lists:** `target_sensors` must be `[["Name", "Type"], ...]`. Never use tuples `()`.
3. **Strict Vocabulary:** - If a user mentions a sensor NOT in `VALID SENSOR NAMES`, mark `is_vague: true`.
   - If `VALID SENSOR NAMES` is "NONE", you cannot accept specific sensor requests.
4. **Acquisition ID Priority:** If `acquisition_id` is extracted, force `condition` and `label_detail` to `null`.

### PARAMETER EXTRACTION LOGIC
1. **Target Sensors:** Extract as `[NAME, TYPE]`. If Type is unspecified, use `null`.
2. **Algorithms:** Map "Logistic Regression"->"lr", "Decision Tree"->"dt", Default->"rf".
3. **Subsets:** "faulty"/"failure" -> "KO"; "healthy"/"normal" -> "OK".

### FEW-SHOT EXAMPLES
User: "Hello, what can you do?"
JSON: {{
  "category": "normal_conversation",
  "is_vague": false,
  "reasoning": "Greeting.",
  "parameters": {{ "visual_config": null, "analysis_config": null }}
}}

User: "Compare Sensor_A and Sensor_B using Logistic Regression."
JSON: {{
  "category": "feature_importance_analysis",
  "is_vague": false,
  "reasoning": "Comparison request.",
  "parameters": {{
      "visual_config": null,
      "analysis_config": {{
         "global": false, 
         "target_sensors": [ ["Sensor_A", null], ["Sensor_B", null] ],
         "algorithm": "lr"
      }}
  }}
}}

User : "frequency spectrum for ISM330DHCX in folder STWIN_00002"
JSON: {{
  "category": "frequency_spectrum",
  "is_vague": false,
  "reasoning": "Specific sensor and folder ID.",
  "parameters": {{
      "analysis_config": null,
      "visual_config": {{
            "target_sensors" : [ ["ISM330DHCX", null] ],
            "subset" : null,
            "condition" : null,
            "label_detail" : null,
            "acquisition_id" : "STWIN_00002"
      }}
  }}
}}
"""
    return PROMPT

# Default fallback (Empty lists)
DEFAULT_ROUTER_PROMPT = build_router_prompt([], [], [], [])


def prepare_responser_prompt(user_query, system_flag=None, tool_output=None, extra_info = None):
    
    responder_input = f"User Query: {user_query}"
    
    # Guidance map for different system flags
    guidance_map = {
        "NORMAL_CONVERSATION": """
Answer the user's question helpfully, BUT ONLY within the scope of IoT and Engineering.
If the user greets you ("Hello"), introduce yourself as an IoT Data Analyst.""",

        "DATA_ANALYSIS_SUCCESS": """
The user's request was valid, and the system has successfully generated statistical data in the Tool Output.
1. ANALYZE the provided 'Tool Output' data.
2. INTERPRET the values for the user (e.g., explain what a high Standard Deviation implies in this context).
3. Do NOT simply output the raw JSON/numbers; provide an engineering summary.""",

        "IRRELEVANT_REQUEST" : """
Do not answer the question at all since it is an out of scope and irrelevant request.
Do not provide opinion, explanations or commentary.""",

        "MISSING_DATASET": """If the system flag is MISSING_DATASET, the user has asked a technical question but has not provided the dataset.
In this case, politely inform the user that they need to first enter the path to their dataset in the left sidebar and click the "Check validity" button.
Once the dataset is successfully provided, they can submit their request again.""",

        "INVALID_SENSORS": """Inform the user that the specific sensors or types they requested could not be found in the processed data.
- Explanation: "The sensors you mentioned are not present in the current dataset or the combination of Name/Type is incorrect."
- Recommendation: Suggest they try a **Global Analysis** (e.g., "Which sensor is best?") or verify the sensor names""",

        "INVALID_ALGORITHM": """Inform the user that the requested model is not supported.
- Explanation: 'I currently support Random Forest (rf), Logistic Regression (lr), and Decision Trees (dt). I cannot perform analysis using [User's Requested Algo].'
- Action: Suggest running the default Random Forest analysis.""",

        "IRRELEVANT_TOPIC": """User Query is about topics unrelated to Industrial IoT, Physics, Engineering, or the specific dataset (e.g., questions about politics, celebrities, cooking, history, general coding):
DO NOT answer the question. DO NOT give opinions on the topic. Reply with a standard refusal""",

        "SUBSET_MISSING": """User query is about a plot generation but the system has found two acquisition folders with the given id
Ask the user to provide the subset (OK/KO) on which they want the requested plot""",

        "INVALID_ACQUSITION": """User query is about a plot generation for a given acquisition folder,
However, the given folder is not found within the provided dataset. Tell the user to carefully write the acquisition name""",

        "MISSING_SENSOR": """The user's request is missing a valid sensor name or the requested sensor
was not found in the dataset. Ask them to specify a valid sensor from the available list.""",

        "BAD_TYPE": """The sensor you specified could not be found. Either the sensor name doesn't exist in the dataset, or the sensor type doesn't match the given sensor name.""",

        "BAD_CONDITION": """The condition was not found. Explain to the user that we search for
conditions by scanning the 'condition' property within the 'metadata.json' files
inside each acquisition folder, and no match was found.""",

        "BAD_LABEL": """The label/fault detail was not found in the dataset. 
Explain that we look for these details in the 'metadata.json' files. 
Suggest that the user verifies their metadata files are complete and correctly labeled.""",

        "BAD_ACQUSITION": """The acquisition folder name was not found in the dataset directory. 
Ask the user to verify the folder name or check if the acquisition exists in the root path.""",

        "TOO_MANY_TARGETS":"""You requested plots for multiple sensor–type pairs.
I have generated the plot for the first requested pair; however, time-series plots can only be created one at a time.
Please submit the remaining requests individually.
        
        """
    }
    
    # Get guidance from map if system_flag exists
    if system_flag and system_flag in guidance_map:
        guidance = guidance_map[system_flag]
        responder_input += f"\nINTERNAL_GUIDANCE: {guidance}\n"
    
    if tool_output:
        responder_input += f"\nTool Output: {tool_output}\n"
    
    return responder_input


