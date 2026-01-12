import os
import json
import glob
import random

def validate_dataset_structure(root_dir):
    """
    Scans the provided root directory to ensure it meets the strict input format requirements.
    
    Checks:
    1. Existence of 'OK' and 'KO' folders.
    2. Existence of session folders inside them.
    3. File consistency (every session must have the exact same set of files).
    4. Naming conventions (SensorName_SensorType.parquet).
    5. Metadata schema validity.

    Returns:
        tuple: (is_ok (bool), reason (str or None))
    """
    
    if not os.path.exists(root_dir):
        return False, f"Root directory does not exist: {root_dir}"

    # --- RULE 1: Check OK and KO folders ---
    required_classes = ["OK", "KO"]
    for cls in required_classes:
        class_path = os.path.join(root_dir, cls)
        if not os.path.isdir(class_path):
            return False, f"Missing required class folder: '{cls}' in root directory."

    # Collect all session folders
    all_sessions = []
    for cls in required_classes:
        class_path = os.path.join(root_dir, cls)
        sessions = [
            os.path.join(class_path, d) for d in os.listdir(class_path) 
            if os.path.isdir(os.path.join(class_path, d))
        ]
        all_sessions.extend(sessions)

    if not all_sessions:
        return False, "No acquisition (session) folders found in OK or KO."

    # --- RULE 4 PREP: Establish the 'Reference' File List ---
    # We pick the first session found as the "Gold Standard". 
    # All other sessions must match this list exactly.
    reference_session = all_sessions[0]
    reference_files = sorted([f for f in os.listdir(reference_session) if not f.startswith('.')])
    
    # Check if reference actually has files
    if not reference_files:
        return False, f"The reference session '{os.path.basename(reference_session)}' is empty."
    
    if "metadata.json" not in reference_files:
        return False, f"Reference session '{os.path.basename(reference_session)}' is missing 'metadata.json'."

    # --- ITERATE EVERY SESSION ---
    for session_path in all_sessions:
        session_name = os.path.basename(session_path)
        current_files = sorted([f for f in os.listdir(session_path) if not f.startswith('.')])

        # --- RULE 4: File Consistency Check ---
        if current_files != reference_files:
            # Find specific difference for helpful error message
            missing = set(reference_files) - set(current_files)
            extra = set(current_files) - set(reference_files)
            return False, (
                f"File mismatch in session '{session_name}'. "
                f"Expected exact match with reference. "
                f"Missing: {missing}, Extra: {extra}"
            )

        # --- RULE 3: File Format & Naming ---
        # We know current_files matches reference_files, so we check content here
        parquet_files = [f for f in current_files if f.endswith(".parquet")]
        
        if not parquet_files:
            return False, f"Session '{session_name}' contains no .parquet files."

        for pq_file in parquet_files:
            # Check naming convention: Name_Type.parquet
            name_no_ext = pq_file.replace(".parquet", "")
            if "_" not in name_no_ext:
                return False, f"Invalid file naming in '{session_name}': '{pq_file}'. Expected 'SensorName_SensorType.parquet'."

        # --- RULE 5: Metadata Schema Validation ---
        meta_path = os.path.join(session_path, "metadata.json")
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except json.JSONDecodeError:
            return False, f"Corrupted JSON in '{session_name}/metadata.json'."
        except Exception as e:
            return False, f"Could not read metadata in '{session_name}': {str(e)}"

        # Check Top-Level Keys
        if "session_info" not in meta or "sensors" not in meta:
            return False, f"Invalid metadata in '{session_name}': Missing 'session_info' or 'sensors' keys."

        # Check Session Info keys
        required_session_keys = ["condition", "fault_detail", "acquisition_id"]
        for k in required_session_keys:
            if k not in meta["session_info"]:
                return False, f"Metadata in '{session_name}' missing session_info key: '{k}'."

        # Check Sensors consistency
        # There must be an entry in 'sensors' for every parquet file found
        meta_sensors = meta["sensors"]
        
        for pq_file in parquet_files:
            # Construct the expected key: "IIS3DWB_ACC.parquet" -> "IIS3DWB_ACC"
            expected_key = pq_file.replace(".parquet", "")
            
            if expected_key not in meta_sensors:
                return False, (
                    f"Metadata mismatch in '{session_name}': "
                    f"File '{pq_file}' exists on disk but is not defined in metadata.json['sensors']."
                )
            
            # Check inner sensor keys
            sensor_data = meta_sensors[expected_key]
            required_sensor_keys = [
                "file_name", "sensor_name", "sensor_type", 
                "units", "columns", "sampling_rate_hz", 
                "is_active", "sensitivity"
            ]
            
            for k in required_sensor_keys:
                if k not in sensor_data:
                    return False, f"Sensor '{expected_key}' in '{session_name}' metadata is missing key: '{k}'."

    # If we survived all loops, the structure is perfect.
    return True, None



def scan_dataset_metadata(root_dir):
    """
    Scans the dataset directory to automatically discover:
    1. Valid Sensor Names (e.g. 'IIS3DWB', 'HTS221')
    2. Valid Sensor Types (e.g. 'ACC', 'TEMP')
    
    It relies on the filename convention: Name_Type.extension
    Works for both Raw (.dat) and Processed (.parquet/.csv) data.
    """
    
    sensor_names = set()
    sensor_types = set()
    conditions = set()
    fault_details = set()
    
    # We look for these extensions to identify data files
    valid_extensions = {'.parquet', '.csv', '.dat', '.json'}

    if not os.path.exists(root_dir):
        return [], []

    # Walk through the entire directory tree
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            
            # 1. Check Extension
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_extensions:
                continue
                
            # 2. Check Naming Convention (Name_Type)
            # Remove extension first
            name_no_ext = os.path.splitext(filename)[0]
            
            # parquet files enter this condition
            if "_" in name_no_ext:
                try:
                    # Split only on the FIRST underscore to be safe
                    # e.g., "ISM330DHCX_ACC" -> ["ISM330DHCX", "ACC"]
                    parts = name_no_ext.split("_")
                    
                    # Basic validation: We expect at least 2 parts
                    if len(parts) >= 2:
                        s_name = parts[0]
                        s_type = parts[1]
                        
                        # Filter out obviously bad names (optional)
                        if len(s_name) > 1 and len(s_type) > 1:
                            sensor_names.add(s_name)
                            sensor_types.add(s_type)
                            
                except Exception:
                    continue # Skip weird files

            if filename == "metadata.json" :
                try :
                    path = os.path.join(root, filename)
                    with open(path,"r") as file:
                        metadata = json.load(file)
                    
                    conditions.add(metadata.get("session_info",{}).get("condition",None))
                    fault_details.add(metadata.get("session_info",{}).get("fault_detail",None))
                except Exception:
                    continue

    return sorted(list(sensor_names)), sorted(list(sensor_types)), sorted(list(conditions)), sorted(list(fault_details))

# this function checks if an acqusition folder exists in the dataset or not
def check_acquisition_presence(root_dir, acqusition_name) :
    found_paths = []
    path_ok = os.path.join(root_dir, "OK", acqusition_name)
    if path_ok : found_paths.append(path_ok)
    path_ko = os.path.join(root_dir, "KO", acqusition_name)
    if path_ko : found_paths.append(path_ko)
    return found_paths


def select_acquisition_paths(root_dir, acquisition_id=None, subset=None, condition=None, label_detail=None):
    """
    Selects a target acquisition folder path based on provided filters.
    
    Logic:
    1. If acquisition_id is provided, search specifically for that folder name.
    2. Otherwise, filter by subset (OK or KO).
    3. For each candidate folder, parse its metadata.json to verify 'condition' and 'fault_detail'.
    4. Return a random folder from the list of valid candidates.
    
    Args:
        root_dir (str): The dataset root (containing OK/KO folders).
        acquisition_id (str): Specific folder name to find.
        subset (str): "OK", "KO", or None.
        condition (str): Condition keyword to match in metadata.
        label_detail (str): Label detail keyword to match in metadata (fault_detail).
        
    Returns:
        str or None: The absolute path to the selected acquisition folder, or None if no match found.
    """
    
    # --- Priority 1: Specific Acquisition ID ---
    if acquisition_id:
        # We search in both OK and KO for the specific folder name
        for category in ["OK", "KO"]:
            category_path = os.path.join(root_dir, category)
            if not os.path.exists(category_path):
                continue
                
            # Check if the folder exists directly in this category
            target = os.path.join(category_path, acquisition_id)
            if os.path.exists(target) and os.path.isdir(target):
                return target
        return None

    # --- Priority 2: Filtered Search via Metadata ---
    
    # Determine which top-level folders to scan
    search_subfolders = []
    if subset == "OK":
        search_subfolders = ["OK"]
    elif subset == "KO":
        search_subfolders = ["KO"]
    else:
        # If subset is None, we look in both
        search_subfolders = ["OK", "KO"]

    candidates = []

    for sub in search_subfolders:
        base_path = os.path.join(root_dir, sub)
        if not os.path.exists(base_path):
            continue

        # List all folders in this category (e.g., all session folders in OK)
        for entry in os.listdir(base_path):
            full_entry_path = os.path.join(base_path, entry)
            
            if os.path.isdir(full_entry_path):
                metadata_path = os.path.join(full_entry_path, "metadata.json")
                
                # If metadata doesn't exist, we can't reliably verify, so we skip
                if not os.path.exists(metadata_path):
                    continue
                
                try:
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                    
                    session_info = meta.get("session_info", {})
                    actual_condition = session_info.get("condition")
                    actual_fault = session_info.get("fault_detail")

                    # Check filters against metadata values
                    matches_condition = True
                    if condition and condition != actual_condition:
                        matches_condition = False
                    
                    matches_label = True
                    if label_detail and label_detail != actual_fault:
                        matches_label = False
                    
                    if matches_condition and matches_label:
                        candidates.append(full_entry_path)
                        
                except (json.JSONDecodeError, IOError, KeyError):
                    # Skip folders with corrupted or missing metadata
                    continue

    # Return one random choice from our candidates
    if candidates:
        return random.choice(candidates)
    
    return None

def is_type_valid (dir , s_name, s_type) :

    path = os.path.join(dir, "metadata.json")

    with open(path, "r") as file:
        j = json.load(file)

    for item in j["sensors"].keys():
        k,v = item.split("_")
        if s_name == k and s_type == v:
            return True

    return False