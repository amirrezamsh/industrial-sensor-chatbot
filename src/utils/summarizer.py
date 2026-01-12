import pandas as pd
import numpy as np
from scipy import stats

def get_signal_summary(df, sensor_name, sensor_type, fs=1.0):
    """
    Converts a large DataFrame of sensor data into a compact 
    dictionary of statistical features for LLM consumption.
    
    Added Metrics:
    - Standard Deviation (Std): Describes the width/thickness of the signal 'fuzz'.
    - Skewness: Describes if the data is leaning or has asymmetric outliers.
    - Slope: Calculated via linear regression to identify rising/falling trends.
    """
    # Ignore Time column for statistical calculation
    data_cols = [c for c in df.columns if c != "Time"]
    
    summary = {
        "sensor": sensor_name,
        "type": sensor_type,
        "sample_count": len(df),
        "features": {}
    }
    
    # Prepare time array for slope calculation
    if "Time" in df.columns:
        time_vals = df["Time"].values
    else:
        # Fallback to index-based time if column is missing
        time_vals = np.arange(len(df)) / fs

    for col in data_cols:
        series = df[col].dropna()
        if series.empty:
            continue
            
        # 1. Calculate Trend (Slope) using Linear Regression
        # x = time, y = sensor values
        slope, _, _, _, _ = stats.linregress(time_vals[:len(series)], series.values)
        
        # 2. Extract key descriptors
        stats_dict = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "max": round(float(series.max()), 4),
            "rms": round(float(np.sqrt(np.mean(series**2))), 4),
            "kurtosis": round(float(series.kurtosis()), 2),
            "skewness": round(float(series.skew()), 2), 
            "slope": round(float(slope), 6), 
            "peak_to_peak": round(float(series.max() - series.min()), 4)
        }
        
        summary["features"][col] = stats_dict
        
    return summary



def format_summary_for_llm(summary):
    """
    Converts the summary dictionary into an enriched string for the LLM.
    Includes natural language hints about trends and shapes.
    """
    text = f"Data Profile for {summary['sensor']} ({summary['type']}):\n"
    for axis, s in summary['features'].items():
        # Trend Descriptor
        if abs(s['slope']) < 0.0001:
            trend_str = "stable/flat"
        else:
            trend_str = "rising" if s['slope'] > 0 else "falling"
            
        # Symmetry Descriptor
        if abs(s['skewness']) < 0.5:
            shape_str = "symmetric"
        else:
            shape_str = "right-leaning" if s['skewness'] > 0 else "left-leaning"

        text += (f"- {axis} Axis: RMS={s['rms']}, Kurtosis={s['kurtosis']}, "
                 f"Std={s['std']} (thickness), Skewness={s['skewness']} ({shape_str}), "
                 f"Slope={s['slope']} ({trend_str})\n")
    return text


#  FREQUENCY SPECTRUM

def get_frequency_summary(frequencies, magnitudes, sensor_name, sensor_type, top_n=3):
    """
    Summarizes the frequency spectrum for LLM interpretation.
    
    Metrics:
    - Peak Frequency: The 'note' the machine is playing loudest.
    - Spectral Centroid: The 'average' frequency (high vs low energy).
    - Top N Peaks: Specific frequencies where energy is concentrated.
    """
    # 1. Find the dominant peak
    peak_idx = np.argmax(magnitudes)
    peak_freq = frequencies[peak_idx]
    peak_mag = magnitudes[peak_idx]
    
    # 2. Spectral Centroid (Center of mass of the spectrum)
    centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes) if np.sum(magnitudes) > 0 else 0
    
    # 3. Find Top N Peaks (to identify harmonics)
    # We sort by magnitude and take the top N
    sort_indices = np.argsort(magnitudes)[::-1]
    top_peaks = []
    for idx in sort_indices[:top_n]:
        top_peaks.append({
            "freq": round(float(frequencies[idx]), 2),
            "mag": round(float(magnitudes[idx]), 4)
        })

    summary = {
        "sensor": sensor_name,
        "type": sensor_type,
        "peak_frequency": round(float(peak_freq), 2),
        "peak_magnitude": round(float(peak_mag), 4),
        "spectral_centroid": round(float(centroid), 2),
        "top_peaks": top_peaks
    }
    
    return summary


def format_freq_summary_for_llm(summary):
    """
    Converts frequency summary into a string for the Responder Agent.
    """
    text = f"Frequency Profile for {summary['sensor']} ({summary['type']}):\n"
    text += f"- Dominant Frequency: {summary['peak_frequency']} Hz (Magnitude: {summary['peak_magnitude']})\n"
    text += f"- Spectral Centroid: {summary['spectral_centroid']} Hz (Overall energy balance)\n"
    text += "- Top Peaks identified:\n"
    for p in summary['top_peaks']:
        text += f"  * {p['freq']} Hz (Mag: {p['mag']})\n"
    return text

