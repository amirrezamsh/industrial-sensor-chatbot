import os
import glob
import pandas as pd
import numpy as np 
from scipy.stats import kurtosis 
import warnings
from concurrent.futures import ProcessPoolExecutor
import json
from src.config import FEATURES_DIR


# --- CONFIGURATION ---
TARGET_SECONDS = 1.0
MASTER_DURATION = 1.23 


def process_sensor_dataframe_vectorized(df, window_size, sensor_name, sensor_type, metadata_dict, odr = 0):
    """
    SPEED UP: 100x faster than looping.
    Converts the DataFrame to a 3D Numpy array and computes stats in one shot.
    """
    
    # 1. Filter Columns (Remove Time and Tags)
    # We only want the data columns (A_x, A_y, A_z)
    data_cols = [c for c in df.columns if c not in ['Time', 'SW_TAG_0', 'SW_TAG_1', 'SW_TAG_2','HW_TAG_0','HW_TAG_1','HW_TAG_2','HW_TAG_3','HW_TAG_4']]
    
    if not data_cols: return pd.DataFrame() # Safety check

    # 2. TRUNCATE DATA
    # We drop the "tail" of the data that doesn't fit into a full window
    n_rows = len(df)
    n_windows = n_rows // window_size
    
    if n_windows == 0: return pd.DataFrame() # File too short for even 1 window

    keep_len = n_windows * window_size
    
    # Extract raw numpy array for just the data columns
    # Shape: (Total_Samples, Num_Axes) e.g., (266670, 3)
    raw_data = df[data_cols].iloc[:keep_len].values 
    
    # 3. RESHAPE TO 3D (The Magic Step)
    # Shape: (Num_Windows, Window_Size, Num_Axes)
    # e.g. (10, 26667, 3)
    reshaped = raw_data.reshape(n_windows, window_size, len(data_cols))
    
    # 4. COMPUTE FEATURES (Vectorized)
    # axis=1 means "crunch along the window dimension"
    
    # Time Domain Stats
    means = np.mean(reshaped, axis=1) # Shape: (10, 3)
    stds  = np.std(reshaped, axis=1)
    peaks = np.max(reshaped, axis=1)
    rmses = np.sqrt(np.mean(reshaped.astype(np.float64)**2, axis=1))
    # Optional: Cast the result back to float32 to save space in the final dataframe
    rmses = rmses.astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        kurts = kurtosis(reshaped, axis=1)
    kurts = np.nan_to_num(kurts, nan=0.0)

    # 5. BUILD RESULT DATAFRAME
    # We need to construct column names matching your format
    feature_data = {}

    # Only run this if the sensor is High Speed (e.g., Vibration/Mic)
    # =========================================================
    if odr > 1000:
        # A. Compute FFT (Vectorized for all windows)
        # rfft = Real FFT (since our data is real numbers, not complex)
        fft_complex = np.fft.rfft(reshaped, axis=1)
        
        # B. Compute Magnitude (Amplitude)
        fft_mag = np.abs(fft_complex)
        
        # C. Get the Frequency Axis (The "X-Axis" labels in Hz)
        # We only need to calculate this once
        freq_labels = np.fft.rfftfreq(window_size, d=1/odr)
        
        # D. Find the Index of the Peak for every window/axis
        # argmax returns the index (0, 1, 2...) of the highest value
        peak_indices = np.argmax(fft_mag, axis=1) # Shape: (n_windows, n_axes)
        
        # E. Convert Index to Hertz
        # We look up the Hz value in freq_labels using the indices
        peak_freqs_hz = freq_labels[peak_indices]
      
  
    # =========================================================
    
    for idx, col_name in enumerate(data_cols):
        clean_col = col_name.split()[0] # Remove units like [g]
        prefix = f'{sensor_name}_{sensor_type}_{clean_col}'
        
        feature_data[f'{prefix}_mean'] = means[:, idx]
        feature_data[f'{prefix}_std']  = stds[:, idx]
        feature_data[f'{prefix}_peak'] = peaks[:, idx]
        feature_data[f'{prefix}_kurtosis'] = kurts[:, idx]
        feature_data[f'{prefix}_rms'] = rmses[:, idx]
        # Frequency Features (Only add if we calculated them)
        if odr > 1000:
            feature_data[f'{prefix}_peak_freq'] = peak_freqs_hz[:, idx]

    # Create the DataFrame
    result_df = pd.DataFrame(feature_data)
    
    # 6. ADD METADATA (Replicated for every window)
    for key, value in metadata_dict.items():
        result_df[key] = value
        
    return result_df


def process_single_acquisition_folder(acq_data_tuple) :
    """
    Worker function to process ONE folder. 
    Returns a dictionary: {'IIS3DWB_ACC': dataframe, 'HTS221_TEMP': dataframe}
    """
    # Unpack arguments (Multiprocessing passes args as a tuple often)
    acq_folder_path, acq_folder_name, binary_label = acq_data_tuple

    with open(os.path.join(acq_folder_path,"metadata.json"),"r") as file :
        acq_info = json.load(file)


    condition_type, fault_detail = acq_info["session_info"]["condition"], acq_info["session_info"]["fault_detail"]

    # print(f"Processing: {acq_folder_name}...")
    
    local_results = {} # Store results for this one folder

    try:

        for file_path in glob.glob(os.path.join(acq_folder_path,"*.parquet")):
                
                file_name = os.path.basename(file_path)
                
                sensor_name = file_name.replace(".parquet","").split("_")[0]
                sensor_type = file_name.replace(".parquet","").split("_")[1]
                odr = acq_info["sensors"][f"{sensor_name}_{sensor_type}"]["sampling_rate_hz"]


                # --- LOAD DATAFRAME ---
                df = pd.read_parquet(file_path)

                if odr is None :
  
                    duration = df['Time'].iloc[-1] - df['Time'].iloc[0]

                    if duration > 0:
                        odr = len(df) / duration
                                
                # --- CALCULATE WINDOW SIZE (Once per sensor) ---
                if odr > 1000: 
                    raw_samples = odr * TARGET_SECONDS
                    window_size = 1 << (int(raw_samples) - 1).bit_length()
                else:
                    window_size = max(1, int(odr * MASTER_DURATION))

                # print("window size is : ",window_size)
                

                # --- VECTORIZED PROCESSING (Replaces the inner loop) ---
                # Prepare metadata to attach to every row
                metadata = {
                    'Condition_Type': condition_type,
                    'Fault_Detail': fault_detail,
                    'Binary_Label': binary_label,
                    'Acquisition_ID': acq_folder_name
                }

                # THIS FUNCTION DOES THE WORK OF THE LOOP
                batch_df = process_sensor_dataframe_vectorized(df, window_size, sensor_name, sensor_type, metadata,odr = odr)
                
                unique_key = f"{sensor_name}_{sensor_type}"
                
                # ADD IT TO THE LOCAL RESULT
                if not batch_df.empty:
                    local_results[unique_key] = batch_df

        # print(f"✅ Done: {acq_folder_name}")

    except Exception as e:
        print(f" ❌ Error in {acq_folder_path}: {e}")

    return local_results


def load_and_label_all_data_parallel(root_dir, dir_name):

    tasks = []

    # 1. Start the root iteration
    for binary_label in os.listdir(root_dir) :
        binary_path = os.path.join(root_dir, binary_label)
        if not os.path.isdir(binary_path): continue

        for acq_folder_name in os.listdir(binary_path) :
            acq_folder_path = os.path.join(binary_path, acq_folder_name)
            if not os.path.isdir(acq_folder_path): continue

            tasks.append((acq_folder_path, acq_folder_name, binary_label))


    print(f"🚀 Starting Parallel Processing on {len(tasks)} folders...")       
    
    final_features_by_sensor = {} # The Bucket Dictionary

    with ProcessPoolExecutor() as executor:
        # This runs the function on all tasks simultaneously
        results = list(executor.map(process_single_acquisition_folder, tasks))

        # C. Merge Results
        # print("Merging results...")
        for result_dict in results:
            for sensor_key, df in result_dict.items():
                if sensor_key not in final_features_by_sensor:
                    final_features_by_sensor[sensor_key] = []
                final_features_by_sensor[sensor_key].append(df)

    # --- FINAL SAVE ---
    # print("\n💾 Saving CSV files...")
    if final_features_by_sensor:

        save_dir = os.path.join(FEATURES_DIR, dir_name)
        # Saving to the directory
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        for name, list_of_dfs in final_features_by_sensor.items():
            # Combine all batches into one big DataFrame
            final_df = pd.concat(list_of_dfs, ignore_index=True)
            
            filename = f"{name}.csv"

            save_path = os.path.join(save_dir, filename)
            
            final_df.to_csv(save_path, index=False)
            # print(f" -> Saved {filename} ({len(final_df)} rows)")
