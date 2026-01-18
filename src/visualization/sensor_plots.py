import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.summarizer import get_signal_summary, format_summary_for_llm, get_frequency_summary, format_freq_summary_for_llm

def get_sensor_visual_report(acquisition_path, sensor_name, sensor_type=None):
    """
    Generates a matplotlib figure for specific sensor data within an acquisition.
    """
    metadata_path = os.path.join(acquisition_path, "metadata.json")
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    sensor_metadata = metadata.get("sensors", {})
    
    # Identify which files/types we are plotting
    target_types = []
    if sensor_type:
        target_types = [sensor_type]
    else:
        # Discover all types available for this specific sensor_name in metadata
        for key, info in sensor_metadata.items():
            if info.get("sensor_name") == sensor_name:
                target_types.append(info.get("sensor_type"))

        targets = [f"{sensor_name}_{tp}.parquet" for tp in target_types]
        path_children = os.listdir(acquisition_path)
        available_paths = list(set(targets).intersection(set(path_children)))
        target_types = [target.split(".")[0].split("_")[1] for target in available_paths]
    
    if not target_types:
        return None

    # Create figure
    fig, axes = plt.subplots(len(target_types), 1, figsize=(10, 4 * len(target_types)), sharex=True)
    if len(target_types) == 1:
        axes = [axes] # Ensure axes is always iterable

    # --- NEW: Extract ID and Set Super Title ---
    # os.path.normpath cleans up mix of slashes, basename gets the final folder name
    acquisition_id = os.path.basename(os.path.normpath(acquisition_path))
    fig.suptitle(f"Acquisition ID: {acquisition_id}", fontsize=14, fontweight='bold', y=0.98)
    # -------------------------------------------

    plot_count = 0
    total_summary = []

    if "OK" in acquisition_path.upper() :
        total_summary.append(f"Analysis for Acquisition: **{acquisition_id}**") # Added to text summary too
        total_summary.append("Condition: OK (Normal State)\n")
    else :
        total_summary.append(f"Analysis for Acquisition: **{acquisition_id}**")
        total_summary.append("Condition: KO (Faulty State)\n")
       
    
    for i, s_type in enumerate(target_types):
        # File naming convention: sensorname_sensortype.parquet
        file_name = f"{sensor_name}_{s_type}.parquet"
        file_path = os.path.join(acquisition_path, file_name)
        
        # Metadata key lookup (e.g., "IIS3DWB_ACC")
        meta_key = f"{sensor_name}_{s_type}"
        info = sensor_metadata.get(meta_key)
        
        if not os.path.exists(file_path) or not info:
            axes[i].text(0.5, 0.5, f"Data missing for {file_name}", ha='center')
            continue

        # Load data
        df = pd.read_parquet(file_path)

        summary = format_summary_for_llm(get_signal_summary(df, sensor_name, s_type))
        total_summary.append(summary)
        
        # Get metadata details
        units = info.get("units", "N/A")
        fs = info.get("sampling_rate_hz", 1.0)
        
        # Retrieve display labels from metadata for the legend
        meta_labels = info.get("columns", [])
        
        # Handle Time data and identify feature columns
        if "Time" in df.columns:
            time_axis = df["Time"]
            data_cols = [c for c in df.columns if c != "Time"]
        else:
            time_axis = [j / fs for j in range(len(df))]
            data_cols = df.columns.tolist()
        
        # Plot each data column directly from the dataframe
        for j, col in enumerate(data_cols):
            label = meta_labels[j] if j < len(meta_labels) else col
            axes[i].plot(time_axis, df[col], label=label, alpha=0.8)
        
        axes[i].set_title(f"Sensor: {sensor_name} | Type: {s_type}")
        axes[i].set_ylabel(f"Value [{units}]")
        axes[i].legend(loc='upper right')
        axes[i].grid(True, which='both', linestyle='--', alpha=0.5)
        plot_count += 1

    axes[-1].set_xlabel("Time [seconds]")
    
    # --- NEW: Layout Adjustment ---
    # 'rect' reserves the top 4% of the figure for the suptitle so it doesn't overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # ------------------------------
    
    return (fig, "\n".join(total_summary)) if plot_count > 0 else None


def get_sensor_frequency_report(acquisition_path, sensor_name, sensor_type=None):
    """
    Generates a frequency spectrum plot (FFT) and a statistical summary for a sensor.
    
    Logic:
    1. Load metadata.json for sampling rates and sensor info.
    2. Load parquet files for the target sensor/type.
    3. Perform FFT on each data column.
    4. Plot Frequency (Hz) vs Magnitude.
    
    Args:
        acquisition_path (str): Path to the acquisition folder.
        sensor_name (str): Name of the sensor (e.g., "IIS3DWB").
        sensor_type (str, optional): Type of data (e.g., "ACC"). Defaults to None.
        
    Returns:
        tuple: (matplotlib.figure.Figure, str) or None: (The plot, the summary text).
    """
    metadata_path = os.path.join(acquisition_path, "metadata.json")
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    sensor_metadata = metadata.get("sensors", {})
    
    # Identify which files/types we are plotting
    target_types = []
    if sensor_type:
        target_types = [sensor_type]
    else:
        for key, info in sensor_metadata.items():
            if info.get("sensor_name") == sensor_name:
                target_types.append(info.get("sensor_type"))

        # this step is necessary since we want to make sure we select those paths that are inside both the acquisition path and metadata
        targets = [f"{sensor_name}_{tp}.parquet" for tp in target_types]
        path_children = os.listdir(acquisition_path)
        available_paths = list(set(targets).intersection(set(path_children)))
        target_types = [target.split(".")[0].split("_")[1] for target in available_paths]
    
    if not target_types:
        return None

    # Create figure
    fig, axes = plt.subplots(len(target_types), 1, figsize=(10, 5 * len(target_types)))
    if len(target_types) == 1:
        axes = [axes]

    plot_count = 0
    total_summary = []
    if "OK" in acquisition_path.upper() :
        total_summary.append("This observation belongs to OK(normal) category")
    else :
        total_summary.append("This observation belongs to KO(faulty) category")
        

    for i, s_type in enumerate(target_types):
        file_name = f"{sensor_name}_{s_type}.parquet"
        file_path = os.path.join(acquisition_path, file_name)
        meta_key = f"{sensor_name}_{s_type}"
        info = sensor_metadata.get(meta_key)
        
        if not os.path.exists(file_path) or not info:
            axes[i].text(0.5, 0.5, f"Data missing for {file_name}", ha='center')
            continue

        # Load data
        df = pd.read_parquet(file_path)
        
        # Get metadata details
        units = info.get("units", "N/A")
        fs = info.get("sampling_rate_hz", 1.0)
        meta_labels = info.get("columns", [])
        
        # Identify data columns (exclude Time)
        data_cols = [c for c in df.columns if c != "Time"]
        
        # FFT Logic
        for j, col in enumerate(data_cols):
            signal = df[col].values
            n = len(signal)
            
            # Remove DC component (mean) to clean up the 0Hz peak
            signal_detrended = signal - np.mean(signal)
            
            # Compute Real FFT
            fft_values = np.fft.rfft(signal_detrended)
            frequencies = np.fft.rfftfreq(n, d=1/fs)
            
            # Calculate Magnitude (normalized by length)
            magnitude = np.abs(fft_values) * 2 / n
            
            label = meta_labels[j] if j < len(meta_labels) else col
            axes[i].plot(frequencies, magnitude, label=label, alpha=0.8)

            # Generate Time-Domain summary for the LLM
            summary = format_freq_summary_for_llm(get_frequency_summary(frequencies, magnitude, sensor_name, s_type))
            total_summary.append(summary)
        
        axes[i].set_title(f"Frequency Spectrum: {sensor_name} ({s_type})")
        axes[i].set_ylabel(f"Magnitude [{units}]")
        axes[i].set_xlabel("Frequency [Hz]")
        axes[i].legend(loc='upper right')
        axes[i].grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Limit x-axis to Nyquist frequency (fs/2)
        axes[i].set_xlim(0, fs/2)
        
        plot_count += 1

    plt.tight_layout()
    
    return (fig, "\n".join(total_summary)) if plot_count > 0 else None