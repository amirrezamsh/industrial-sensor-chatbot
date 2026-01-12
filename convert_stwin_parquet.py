import os
import glob
import json
import pandas as pd
import datetime
import shutil
from HSD.HSDatalog import HSDatalog
import warnings
import numpy as np

# Configuration: Where to save the clean data
OUTPUT_ROOT = r"D:\semester 4\system and device\project\AI-chatbot\data\clean" 

def get_file_timestamp(file_path):
    """Extracts file creation time for metadata."""
    try:
        timestamp_float = os.path.getmtime(file_path)
        dt_object = datetime.datetime.fromtimestamp(timestamp_float)
        return dt_object.isoformat()
    except Exception:
        return None

def read_json_metadata(acq_folder):
    """
    Reads DeviceConfig.json to extract physics/context for the sensors.
    Returns a dictionary structured for the final metadata.json.
    """
    path = os.path.join(acq_folder, "DeviceConfig.json")
    
    if not os.path.exists(path):
        print(f"⚠️ Warning: No DeviceConfig.json found in {acq_folder}")
        return {}

    with open(path, "r") as file:
        data = json.load(file)
    
    sensors_dict = {}

    # The key is usually "sensor" (singular) in STWIN files, but checking both just in case
    sensor_list = data.get("device",{}).get("sensor", data.get("sensors", []))

    for sensor in sensor_list:
        s_name = sensor["name"] # e.g. "IIS3DWB"

        desc_list = sensor["sensorDescriptor"]["subSensorDescriptor"]
        status_list = sensor["sensorStatus"]["subSensorStatus"]

        # Iterate through sub-sensors (ACC, GYRO, etc.)
        for desc, status in zip(desc_list, status_list):
            
            
            s_type = desc["sensorType"]       # "ACC"
            s_unit = desc["unit"]             # "g"
            s_cols = desc["dimensionsLabel"]  # ["x", "y", "z"]
            
            s_odr = status.get("ODRMeasured",None)     # 26667.0
            s_active = status["isActive"]     # true/false
            s_sens = status["sensitivity"]    # 0.000488

            # Create a unique key for the dictionary
            key = f"{s_name}_{s_type}"
            
            sensors_dict[key] = {
                "file_name": f"{key}.csv",
                "sensor_name": s_name,
                "sensor_type": s_type,
                "units": s_unit,
                "columns": s_cols,
                "sampling_rate_hz": s_odr,
                "is_active": s_active,
                "sensitivity": s_sens
            }
            
    return sensors_dict

def convert_data(root_dir=r"D:\semester 4\system and device\project\AI-chatbot\data\Sensor_STWIN"):
    
    print(f"🚀 Starting conversion from: {root_dir}")
    print(f"📂 Output will be saved to: {os.path.abspath(OUTPUT_ROOT)}\n")

    # 1. Walk the directory tree
    if not os.path.exists(root_dir):
        print("Error: Source path does not exist.")
        return

    # Structure: Root -> Condition (vel-fissa) -> Fault (KO_HIGH) -> Acquisition (PMI_100)
    for condition in os.listdir(root_dir):
        condition_path = os.path.join(root_dir, condition)
        if not os.path.isdir(condition_path): continue

        for fault_type in os.listdir(condition_path):
            fault_path = os.path.join(condition_path, fault_type)
            if not os.path.isdir(fault_path): continue

            # Determine the Class Label (OK vs KO) based on folder name
            class_label = "OK" if "OK" in fault_type else "KO"
            
            for acquisition in os.listdir(fault_path):
                acq_path = os.path.join(fault_path, acquisition)
                if not os.path.isdir(acq_path): continue

                print(f"Processing: {condition} | {fault_type} | {acquisition}")

                # --- STEP A: PREPARE DESTINATION ---
                # Create the standardized session name
                # e.g. "vel-fissa_KO-HIGH-2mm_PMS-100rpm"
                session_name = f"{condition}_{fault_type}_{acquisition}"
                
                # Final path: converted_data/KO/vel-fissa_.../
                dest_folder = os.path.join(OUTPUT_ROOT, class_label, session_name)
                
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)

                # --- STEP B: READ METADATA ---
                sensor_meta = read_json_metadata(acq_path)
                
                # Construct the full metadata.json content
                full_metadata = {
                    "session_info": {
                        "condition": condition,
                        "fault_detail": fault_type,
                        "acquisition_id": acquisition
                    },
                    "sensors": sensor_meta
                }

                # Save metadata.json
                with open(os.path.join(dest_folder, "metadata.json"), "w") as f:
                    json.dump(full_metadata, f, indent=4)

                # --- STEP C: CONVERT DATA FILES ---
                # Initialize HSDatalog for this folder
                try:
                    hsd = HSDatalog(acquisition_folder=acq_path)
                    
                    # Find all .dat files to convert
                    dat_files = glob.glob(os.path.join(acq_path, "*.dat"))
                    
                    for dat_file in dat_files:
                        # Parse filename: IIS3DWB_ACC.dat -> Name: IIS3DWB, Type: ACC
                        # Windows path split might need \\ or / depending on OS, os.path.basename is safer
                        filename = os.path.basename(dat_file)
                        name_no_ext = filename.replace(".dat", "")
                        
                        if "_" not in name_no_ext: continue
                        
                        parts = name_no_ext.split("_")
                        s_name = parts[0]
                        s_type = parts[1]
                        
                        # Check if this sensor is in our metadata (and active)
                        meta_key = f"{s_name}_{s_type}"
                        if meta_key in sensor_meta and not sensor_meta[meta_key]['is_active']:
                            continue # Skip inactive sensors

                        # Load DataFrame using SDK
                        try:

                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")

                                df = hsd.get_dataframe(
                                    sensor_name=s_name, 
                                    sub_sensor_type=s_type, 
                                    labeled=False
                                )
                            
                            if df is not None and not df.empty:

                                # --- THE REPAIR LOGIC ---
                                # Check if 'Time' is broken BUT data columns are okay
                                data_cols = [c for c in df.columns if c != 'Time']
                                
                                time_has_nans = df['Time'].isnull().any()
                                data_is_good = not df[data_cols].isnull().any().any() 

                                if time_has_nans and data_is_good:
                                    print(f"   🔧 Repairing Time Column for {filename}...")
                                    
                                    # Get the Sampling Rate (ODR) from your metadata logic earlier
                                    # (Assuming you extracted 's_odr' from the JSON)
                                    # If you don't have it handy, estimate it from the valid time data or use theoretical
                                    s_odr = sensor_meta[f"{s_name}_{s_type}"]["sampling_rate_hz"]
                                    if 'sampling_rate_hz' in sensor_meta[f"{s_name}_{s_type}"].keys() and s_odr > 0:
                                        odr_to_use = s_odr
                                    else:
                                        # Fallback: Estimate ODR from the valid parts of the file (average step)
                                        valid_time = df['Time'].dropna()
                                        if len(valid_time) > 1:
                                            duration = valid_time.iloc[-1] - valid_time.iloc[0]
                                            count = len(valid_time)
                                            odr_to_use = count / duration
                                        else:
                                            print("   ❌ Cannot repair: ODR unknown.")
                                            continue

                                    # REGENERATE TIME COLUMN
                                    # Time = Start + (Index / ODR)
                                    start_time = df['Time'].dropna().iloc[0] if not df['Time'].dropna().empty else 0.0
                                    
                                    # Create new time array: 0, 1, 2... divided by ODR
                                    new_time = np.arange(len(df)) / odr_to_use
                                    
                                    # Apply offset
                                    df['Time'] = start_time + new_time
                                    
                                    print("   ✅ Repair Successful.")

                                elif df.isnull().values.any():
                                    # If Data columns (G_x) are also null, then the file is truly broken.
                                    print(f"   ⚠️ Skipping {filename}: Sensor Data is Corrupt.")
                                    continue

                                # 3. Proceed to save...

                                # 🟢 OPTIMIZATION : Downcast Floats to 32-bit
                                # This cuts memory usage in half immediately
                                float_cols = df.select_dtypes(include=['float64']).columns
                                cols_to_shrink = [c for c in float_cols if 'Time' not in c]

                                df[cols_to_shrink] = df[cols_to_shrink].astype('float32')

                                # Save as parquet
                                parquet_name = f"{s_name}_{s_type}.parquet"
                                save_path = os.path.join(dest_folder, parquet_name)
                                df.to_parquet(save_path, index=False, compression='zstd')
                                
                        except Exception as e:
                            print(f"   ❌ Failed to convert {filename}: {e}")

                except Exception as e:
                    print(f"   ❌ Failed to initialize HSDatalog: {e}")

    print("\n✅ Conversion Complete!")

if __name__ == "__main__":
    convert_data()