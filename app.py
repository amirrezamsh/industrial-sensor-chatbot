import streamlit as st
import pandas as pd
import datetime
import time
from src.agent.core import check_ollama_connection
from src.agent.core import generate_ollma_response
from src.utils.file_utils import validate_dataset_structure, scan_dataset_metadata
from src.agent.prompts import build_router_prompt, build_responder_prompt
from src.config import FEATURES_DIR

import time
import os


@st.dialog("Similar folder found")
def vote():
    st.write(f"I have found an analysis for a folder with the same name, Do you want me to compute features again?")
    if st.button("No", type="primary") :
        st.session_state.FEATURES_PATH = os.path.basename(st.session_state.DATASET_PATH)
        st.rerun()
    elif st.button("Yes"):
        st.session_state.FEATURES_PATH = f"{os.path.basename(st.session_state.DATASET_PATH)}_{int(time.time())}"
        st.rerun()


#--- CONFIGURATION ---
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=3)
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama API endpoint
OLLAMA_MODEL = "llama3.1:8b-instruct-q3_K_M"

SUGGESTIONS = {
    "Global feature importance analysis": (
        "Which features in my entire dataset better separate the faulty samples from the normal ones?"
    ),
    "Metadata of dataset": (
        "what are the different sesnors in my dataset? what are the different sensor types? Also provide me with different fault details and conditions in the dataset"
    ),
    "Ask about my capabilities":(
        "Can you tell me what things you are able to do?"
    )
}

is_ok = False

# This ensures the variable exists before we try to use it anywhere
if "DATASET_PATH" not in st.session_state:
    st.session_state.DATASET_PATH = None

if "ROUTER_PROMPT" not in st.session_state :
    st.session_state.ROUTER_PROMPT = build_router_prompt([], [], [], [])

if "RESPONDER_PROMPT" not in st.session_state :
    st.session_state.RESPONDER_PROMPT = build_responder_prompt([], [], [], [])

if "FEATURES_PATH" not in st.session_state:
    st.session_state.FEATURES_PATH = None


if not check_ollama_connection() :
    st.error("ollama is not connected")
    st.stop()


title_row = st.container(
    horizontal=True,
    vertical_alignment="bottom",
)

with title_row:
    st.title(
        # ":material/cognition_2: Streamlit AI assistant", anchor=False, width="stretch"
        "SenseTime AI",
        anchor=False,
        width="stretch",
    )


dataset_path = st.sidebar.text_input(
    "Enter local dataset path:", 
    value=r"D:\semester 4\system and device\project\DATA\clean"
)


# 2. Validate the Path
if st.sidebar.button("Chcek validity"):
    if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
        is_ok, reason = validate_dataset_structure(dataset_path)
        if is_ok :
            st.session_state.validation_status = "success"
            st.session_state.validation_message = "✅ Input data is valid"
            st.session_state.DATASET_PATH = dataset_path
            sensor_names, sensor_types, conditions, faults_details = scan_dataset_metadata(st.session_state.DATASET_PATH)
            st.session_state.ROUTER_PROMPT = build_router_prompt(sensor_names, sensor_types, conditions, faults_details)
            st.session_state.RESPONDER_PROMPT = build_responder_prompt(sensor_names, sensor_types, conditions, faults_details)

            if os.path.basename(st.session_state.DATASET_PATH) in os.listdir(FEATURES_DIR) :
                vote()
            else :
                st.session_state.FEATURES_PATH = os.path.basename(st.session_state.DATASET_PATH)
        else :

            st.session_state.validation_status = "error"
            st.session_state.validation_message = f"❌ Oops : {reason}"
            st.session_state.DATASET_PATH = None
    else :
            st.session_state.validation_status = "error"
            st.session_state.validation_message = f"❌ Oops, take a look at your path again"
            st.session_state.DATASET_PATH = None


if "validation_status" in st.session_state:
    if st.session_state.validation_status == "success":
        st.sidebar.success(st.session_state.validation_message)
    elif st.session_state.validation_status == "error":
        st.sidebar.error(st.session_state.validation_message)



user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)

user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)

user_first_interaction = (
    user_just_asked_initial_question or user_just_clicked_suggestion
)

has_message_history = (
    "messages" in st.session_state and len(st.session_state.messages) > 0
)



# Show a different UI when the user hasn't asked a question yet.
if not user_first_interaction and not has_message_history:
    st.session_state.messages = []

    with st.container():
        st.chat_input("Ask a question...", key="initial_question")

        selected_suggestion = st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
        )

        st.stop()      # Stop here until user chooses something


user_message = st.chat_input("Ask a follow-up...")

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]


with title_row:

    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None
        st.session_state.DATASET_PATH = None
        st.session_state.ROUTER_PROMPT = build_router_prompt([],[], [], [])
        st.session_state.RESPONDER_PROMPT = build_router_prompt([],[], [], [])
        if "validation_status" in st.session_state:
            del st.session_state["validation_status"]
        if "validation_message" in st.session_state:
            del st.session_state["validation_message"]
        


    st.button(
        "Restart",
        icon=":material/refresh:",
        on_click=clear_conversation,
    )

if "prev_question_timestamp" not in st.session_state:
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

# Display chat messages from history as speech bubbles.
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if "images" in message and message["images"]:
            for fig in message["images"]:
                st.pyplot(fig, width="stretch")

        if message["role"] == "assistant":
            st.container()  # Fix ghost message bug.

        st.markdown(message["content"])

if user_message:

    # 1. IMMEDIATE STATE UPDATE
    # Add the user message to history NOW. This prevents UI sync issues if
    # the app re-runs while processing.
    st.session_state.messages.append({"role": "user", "content": user_message})

    # -------------------------------------------------------------------------
    # 2. IMMEDIATE RENDER
    # Display the user message bubble immediately.
    # -------------------------------------------------------------------------
    user_message_clean = user_message.replace("$", r"\$")
    with st.chat_message("user"):
        st.text(user_message_clean)



    # Display assistant response as a speech bubble.
    with st.chat_message("assistant"):
        with st.spinner("Waiting..."):
            # Rate-limit the input if needed.
            question_timestamp = datetime.datetime.now()
            time_diff = question_timestamp - st.session_state.prev_question_timestamp
            st.session_state.prev_question_timestamp = question_timestamp

            if time_diff < MIN_TIME_BETWEEN_REQUESTS:
                time.sleep(time_diff.seconds + time_diff.microseconds * 0.001)

            
            # CRITICAL FIX: Pass a COPY of the messages to the function.
            # Use .copy() or list slicing [:] so the function doesn't modify the global state.
            history_copy = st.session_state.messages.copy() 
            response_generator = generate_ollma_response(user_message, history_copy)

        with st.container():
            # Stream the LLM response.
            response_text = ""
            response_figures = []
            #Create a dedicated placeholder for Status Updates
            status_placeholder = st.empty()

            placeholder = st.empty()

            for chunk in response_generator:

                if isinstance(chunk, tuple):

                    # CHECK: Is this chunk actually A STATUS TO BE SHOWN TO USER?
                    if chunk[0] == "__STATUS__" :
                        status_msg = chunk[1]
                        if status_msg == "COMPLETE":
                            status_placeholder.empty() # Remove the status message
                        else:
                            # Show a nice spinner-like text
                            status_placeholder.markdown(f"*{status_msg}*") 
                        continue

                    # CHECK: Is this chunk actually our list of figures?
                    elif chunk[0] == "__FIGURES__" :
                        response_figures = chunk[1]
                        continue # Don't print this, just store it
                    

                response_text += chunk
                placeholder.markdown(response_text + "▌")
                time.sleep(0.01)

            placeholder.markdown(response_text)  # final render without cursor

            # **TEMPORARY CHANGE:** Skip process_ai_command and treat as final response
            final_response_output = response_text

            # -----------------------------------------------------------------
            # 4. FINAL STATE UPDATE
            # Only append the ASSISTANT response here. 
            # (The user message was already appended at step 1).
            # -----------------------------------------------------------------
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "images": response_figures
            })


