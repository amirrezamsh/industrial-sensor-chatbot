import os
import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
from datetime import timedelta
from src.config import OLLAMA_MODEL, OLLAMA_API_URL, RESPONDER_AGENT_PROMPT, ALGORITHMS, FEATURES_DIR
from src.analysis.feature_importance import run_analysis, generate_summary_string
from src.analysis.feature_extractor import load_and_label_all_data_parallel
from src.utils.file_utils import check_acquisition_presence, select_acquisition_paths
from src.agent.prompts import prepare_responser_prompt
from src.visualization.sensor_plots import get_sensor_visual_report, get_sensor_frequency_report
from src.utils.file_utils import scan_dataset_metadata, is_type_valid

# --- LLM COMMUNICATION AND INTENT CLASSIFICATION ---

@st.cache_data(ttl=timedelta(minutes=5))
def check_ollama_connection():
    """Checks if the local Ollama server is running and accessible."""
    try:
        response = requests.get(OLLAMA_API_URL.replace("/chat", "/tags"), timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.RequestException:
        return False

def router_agent(prompt, message_history):
    """
    Sends the prompt to the local Ollama server and streams/returns the response.
    The response format is handled by the SYSTEM_PROMPT defined in config.py.
    """
    
    # 1. Prepare messages for Llama 3.1 (System + History + User)
    messages = [{"role": "system", "content": st.session_state.ROUTER_PROMPT}]
    for message in message_history:
        # Use content and role from Streamlit session state history
        messages.append({"role": message["role"], "content": message["content"]})
        
    messages.append({"role": "user", "content": prompt})

    # 2. Define the payload
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False, # Disable streaming
        "format": "json" # Request JSON for command intents
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()

        response_json = response.json()
        content = response_json.get("message",{}).get("content","")
        return content

    except requests.exceptions.RequestException as e:
        st.error(f"Ollama Connection Error: Is the server running and is the model '{OLLAMA_MODEL}' loaded? Error: {e}")
        return ""
    

def responder_agent(prompt, message_history):
    # 1. Prepare messages for Llama 3.1 (System + History + User)
    messages = [{"role": "system", "content": RESPONDER_AGENT_PROMPT}]
    for message in message_history:
        # Use content and role from Streamlit session state history
        messages.append({"role": message["role"], "content": message["content"]})
        
    messages.append({"role": "user", "content": prompt})

    # 2. Define the payload
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True, # Enable streaming
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()

        full_response = ""

        for line in response.iter_lines(chunk_size=1) :
            if line :
                try :
                    chunk = json.loads(line)
                    content = chunk.get("message",{}).get("content","")

                    if content :
                        full_response += content
                        yield content
                    
                    if chunk.get("done") :
                        break
                except json.JSONDecodeError :
                    continue

    except requests.exceptions.RequestException as e:
        st.error(f"Ollama Connection Error: Is the server running and is the model '{OLLAMA_MODEL}' loaded? Error: {e}")
        return ""



def generate_ollma_response(prompt, message_history) :
    yield ("__STATUS__", "🧠 Routing your request...")
    router_agent_response = router_agent(prompt, message_history)


    generated_figures = [] # Store figures here

    tool_output = ""
    system_flag = ""

    try:
        req = json.loads(router_agent_response)

        req_category = req.get("category", "normal_conversation")
        is_vague = req.get("is_vague", False)
        reason = req.get("reasoning", None)

        # Extract new parameters
        if req_category == "feature_importance_analysis" :
            params = req.get("parameters", {}).get("analysis_config",{})
            is_global = params.get("global", True)
            target_sensors = params.get("target_sensors",[])
            alg = params.get("algorithm","rf")

        elif req_category == "time_series" or req_category == "frequency_spectrum"  :
            params = req.get("parameters", {}).get("visual_config",{})
            target_sensors = params.get("target_sensors",[])

            if target_sensors :
                if len(set([item[0] for item in target_sensors])) == 1 and len(target_sensors) > 1:
                    sensor = target_sensors[0][0]
                    sensor_type = None
                else :
                    sensor, sensor_type = target_sensors[0]
                    if len(target_sensors) > 1:
                        system_flag = "TOO_MANY_TARGETS"

            else :
                sensor = sensor_type = None

            subset = params.get("subset", None)
            condition = params.get("condition", None)
            label_detail = params.get("label_detail", None)
            acqusition_id = params.get("acquisition_id", None)
        
        print(req,end="\n\n")

    except json.JSONDecodeError:
        req_category = "normal_conversation"
        is_vague = False


    if req_category == "feature_importance_analysis" :
        if not st.session_state.DATASET_PATH :
            system_flag = "MISSING_DATASET"
        else:
            
            if is_vague :
                system_flag = "VAGUE"

            elif alg == "unsupported" or alg not in ALGORITHMS.keys() :
                system_flag = "INVALID_ALGORITHM"

            else :
                # 2. Send "Heavy Work" status
                data_path = os.path.join(FEATURES_DIR, st.session_state.FEATURES_PATH)
                if not os.path.exists(data_path) or len(os.listdir(data_path)) < 1 :
                    yield ("__STATUS__", "⚙️ Extracting features from your data, please be patient ...")
                    load_and_label_all_data_parallel(st.session_state.DATASET_PATH, st.session_state.FEATURES_PATH)

                yield ("__STATUS__", f"⚙️ Running {ALGORITHMS[alg]}, this might take few seconds ...")
                is_valid, results = run_analysis(data_folder=data_path, algorithm= alg, target_sensors = target_sensors)

                if is_valid :
                    sensor_scores, top_features, ax1, ax2 = results
                    yield ("__STATUS__", "✅ Analysis Complete. Generating response...")
                    summary_string = generate_summary_string(sensor_scores, top_features)
                    tool_output = summary_string

                    st.pyplot(ax1.figure, width= "stretch")
                    st.pyplot(ax2.figure, width="stretch")

                    generated_figures.extend([ax1.figure, ax2.figure])
                else :
                    yield ("__STATUS__", "❌ Failed to do the analysis ...")
                    system_flag = "INVALID_SENSORS"


    elif req_category == "data_visualization" or  req_category == "frequency_spectrum":
        if not st.session_state.DATASET_PATH :
            system_flag = "MISSING_DATASET"
        else :
            sensors, _ , conditions, label_details = scan_dataset_metadata(st.session_state.DATASET_PATH)

            if sensor not in sensors :
                system_flag = "MISSING_SENSOR"

            elif condition and condition not in conditions :
                system_flag = "BAD_CONDITION"

            elif label_detail and label_detail not in label_details :
                system_flag = "BAD_LABEL"
            else :

                founded_path = select_acquisition_paths(st.session_state.DATASET_PATH ,acqusition_id, subset, condition, label_detail)
                if founded_path :

                    if sensor_type and not is_type_valid(founded_path, sensor, sensor_type) :
                        system_flag = "BAD_TYPE"
                    else :

                        if req_category == "data_visualization":
                            fig, tool_output = get_sensor_visual_report(
                                acquisition_path= founded_path,
                                sensor_name= sensor,
                                sensor_type= sensor_type
                            )
                        else :
                            fig, tool_output = get_sensor_frequency_report(
                                acquisition_path= founded_path,
                                sensor_name= sensor,
                                sensor_type= sensor_type
                            )

                        st.pyplot(fig, width="stretch")
                        generated_figures.extend([fig])
                else :
                    system_flag = "BAD_ACQUSITION"


    elif req_category == "irrelevant_request" :
        system_flag = "IRRELEVANT_REQUEST"
                        
    elif req_category == "normal_conversation" :
        system_flag = "NORMAL_CONVERSATION"

    if (req_category == "feature_importance_analysis" or req_category == "time_series" or req_category == "frequency_spectrum") and not system_flag:
        system_flag = "DATA_ANALYSIS_SUCCESS"


    responder_input = prepare_responser_prompt(user_query= prompt, system_flag= system_flag, tool_output= tool_output)

    # 4. Clear status before speaking (Optional, essentially "Ready")
    yield ("__STATUS__", "COMPLETE")
    
    yield ("__FIGURES__", generated_figures)
    yield from responder_agent(responder_input, message_history)

