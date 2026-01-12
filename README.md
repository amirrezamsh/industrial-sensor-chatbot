# Industrial IoT Predictive Maintenance Chatbot 🤖

An AI-powered conversational agent designed to analyze, visualize, and interpret high-frequency time-series data from industrial sensors. This tool bridges the gap between raw sensor data and actionable engineering insights using a local LLM (Llama 3) and Machine Learning analysis.

**Note:** This agent is hardware-agnostic. It works with any time-series dataset that follows the standard input structure (Parquet/CSV). It does not accept raw binary files directly.

## 📋 Features

- **Intelligent Routing:** Automatically categorizes user intents (Analysis vs. Visualization vs. General Chat) using a Router Agent.

- **Feature Importance Analysis:** Trains Random Forest models on-the-fly to identify which sensors and statistical features (Mean, Kurtosis, FFT Peak) best predict machine faults.

- **Advanced Visualization:** Generates Time-Series and Frequency Domain (FFT) plots on demand.

- **Data Summarization:** Translates complex signal statistics into natural language explanations using a Responder Agent.

- **Local-First Architecture:** Processes GBs of data locally for privacy and speed, using a local Ollama LLM.

## 🛠️ Requirements & Prerequisites

### 1. Python Environment

- This project requires **Python 3.9+**.
- All dependencies are listed in `requirements.txt`.

### 2. Local LLM Setup (Ollama)

This chatbot uses a local Large Language Model to ensure data privacy and zero latency. You must have **Ollama** installed and running.

1. **Download Ollama:** Visit [ollama.com](https://ollama.com) and install it for your OS.

2. **Pull the Model:** Open your terminal and run:

```bash
   ollama pull llama3
```

(Note: You can change the model in `src/config.py` if you prefer `mistral` or others).

3. **Start the Server:** Ensure Ollama is running in the background (usually on `http://localhost:11434`).

## 🚀 Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/amirrezamsh/industrial-sensor-chatbot.git
cd industrial-iot-chatbot
```

### Install Python Libraries

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### Optional: Configure SDK for STWIN Data

**Only required if you need to convert raw STWIN binary files.**

If you are using raw data from the STMicroelectronics STWIN kit, this project provides a converter script that relies on the **HSDatalog SDK**.

- Ensure the `HSD` folder is located inside `external_libs/`.
- If you are using a different SDK version, update the path in `src/config.py`.

## 📂 Input Data Format

The agent is designed to be **General** and adaptable to different machines. It does not accept raw binary files directly. Instead, your input data must follow a strict **Class-Session** folder structure with standard file formats.

### 1. Directory Structure

The root folder of your dataset must be organized by **Classification Label** (e.g., `OK` vs `KO`). Inside each class folder, each sub-folder represents one **Recording Session**.

```
/My_Dataset_Root
│
├── /OK  (Healthy Data)
│   ├── /Session_Run_001/
│   │   ├── metadata.json
│   │   ├── IIS3DWB_ACC.parquet
│   │   └── HTS221_TEMP.parquet
│   └── ...
│
└── /KO  (Faulty Data)
    ├── /Session_Run_002/
    │   ├── metadata.json
    │   ├── IIS3DWB_ACC.parquet
    │   └── ...
    └── ...
```

### 2. File Requirements

- **Format:** Files must be in **Parquet** (`.parquet`) or **CSV** (`.csv`) format.

- **Naming:** Files must follow the convention: `SensorName_SensorType.parquet` (e.g., `IIS3DWB_ACC.parquet`).

- **Content:** Must include a `Time` column (seconds) and data columns (e.g., `x`, `y`, `z`).

### 3. Metadata (Crucial)

Every session folder must contain a `metadata.json` file describing the sensors. This allows the agent to interpret units and sampling rates correctly.

```json
{
  "session_info": {
    "condition": "vel-fissa",
    "fault_detail": "KO_HIGH_2mm"
  },
  "sensors": {
    "IIS3DWB_ACC": {
      "sensor_name": "IIS3DWB",
      "sensor_type": "ACC",
      "units": "g",
      "sampling_rate_hz": 26667.0
    }
  }
}
```

### 4. For STWIN Users (Raw Binary Converter)

If you have raw STWIN data (`.dat` binary files), you can use the provided helper script to transform them into the standard Parquet format required by the agent:

```bash
python convert_stwin_parquet.py
```

This script will read the proprietary binary structure using the HSD SDK and generate the clean `OK/KO` folder structure described above.

## 💻 How to Run the Application

1. **Start Ollama:** Ensure your local LLM server is active.

2. **Run the Streamlit App:**

```bash
   streamlit run app.py
```

3. **Interact:**

   - Open your browser to `http://localhost:8501`.

   - **Sidebar:** Enter the path to your dataset folder (e.g., `D:\Data\My_Clean_Data`).

   - **Chat:** Ask questions like:

     - "Which sensor is best at detecting faults?"

     - "Show me the frequency spectrum for the Vibration sensor."

     - "Compare the time series of Healthy vs Faulty data."

## 🧠 System Architecture

The chatbot operates on a **Router-Responder** architecture:

- **Router Agent:** Analyzes your prompt and classifies it (e.g., `feature_importance_analysis`, `data_visualization`). It extracts parameters like "sensor name" or "algorithm" into a JSON object.

- **Backend Logic:** Executes Python code based on the intent:

  - **Analysis:** Runs Random Forest / Logistic Regression on the data.

  - **Visualization:** Generates Matplotlib figures from Parquet files.

- **Responder Agent:** Receives the raw tool output (charts, stats) and generates a professional, engineering-focused explanation for the user.
