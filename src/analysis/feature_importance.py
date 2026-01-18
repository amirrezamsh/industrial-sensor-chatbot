import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def get_valid_sensor_files(data_folder, target_sensors):
    """
    Validates user requests against actual files on disk.
    
    Args:
        data_folder (str): Path to processed CSVs.
        target_sensors (list): List of requests [['NAME', 'TYPE'], [None, 'TYPE']].
                               Empty list [] implies Global Analysis.

    Returns:
        tuple: (all_correct (bool), valid_paths (list))
    """
    
    # 1. Catalog all available files into a lookup dictionary
    # Structure: available_map['SENSOR_NAME']['SENSOR_TYPE'] = 'full/path/to/file.csv'
    available_map = {}
    
    # Get all csv files
    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    # If Global Analysis (empty target list), return everything immediately
    if not target_sensors:
        return True, all_files

    # Build the Index
    for file_path in all_files:
        filename = os.path.basename(file_path).replace(".csv", "")
        
        # Valid format: NAME_TYPE.csv
        if "_" not in filename: continue
            
        s_name, s_type = filename.split("_", 1)
        
        if s_name not in available_map:
            available_map[s_name] = {}
        
        available_map[s_name][s_type] = file_path

    # 2. Validate Requests
    valid_paths = []
    all_correct = True
    
    for req in target_sensors:
        req_name = req[0]
        req_type = req[1] # Can be None
        
        # --- NEW LOGIC START ---
        
        # CASE 1: Name is NONE (User wants specific Type across ALL sensors)
        # Example: [None, 'TEMP'] -> should fetch HTS221_TEMP, LPS22HH_TEMP, STTS751_TEMP
        if req_name is None:
            if req_type is None:
                # [None, None] technically implies "All files" (rarely passed explicitly)
                for s in available_map:
                    valid_paths.extend(available_map[s].values())
            else:
                found_type_anywhere = False
                # Iterate through all known sensors to find this type
                for s_key in available_map:
                    if req_type in available_map[s_key]:
                        valid_paths.append(available_map[s_key][req_type])
                        found_type_anywhere = True
                
                # If we went through all sensors and didn't find this type anywhere, it's an error
                if not found_type_anywhere:
                    all_correct = False
        
        # CASE 2: Specific Sensor Name provided
        else:
            # Check if Sensor Name exists
            if req_name not in available_map:
                all_correct = False
                continue
            
            # Sub-case: Name + Any Type ([ 'HTS221', None ])
            if req_type is None:
                paths = list(available_map[req_name].values())
                valid_paths.extend(paths)
                
            # Sub-case: Name + Specific Type ([ 'HTS221', 'HUM' ])
            else:
                if req_type in available_map[req_name]:
                    valid_paths.append(available_map[req_name][req_type])
                else:
                    all_correct = False
                    
        # --- NEW LOGIC END ---

    # Remove duplicates just in case
    valid_paths = list(set(valid_paths))
    
    return all_correct, valid_paths


def run_analysis(data_folder, algorithm = "rf", target_sensors = []):
    """
    Reads all sensor CSVs in the folder and performs feature importance analysis.
    """
    
    # List to store global results
    global_ranking = []
    csv_files = []
    
    # 1. Find all valid CSV files (e.g., IIS3DWB_ACC.csv, HTS221_TEMP.csv)
    is_valid, csv_files = get_valid_sensor_files(data_folder , target_sensors)

    if not is_valid:
        print("The requested sensors are not valid - check again!")
        return (False , ())

    for file_path in csv_files:
        sensor_name = os.path.basename(file_path).replace(".csv", "")
        
        # 2. Load Data
        df = pd.read_csv(file_path)
        
        # 3. Prepare X (Features) and y (Labels)
        # Drop metadata columns to isolate the features
        ignore_cols = ['Condition_Type', 'Fault_Detail', 'Binary_Label', 
                       'Acquisition_ID', 'Window_Start_Index', 'Sensor_Name']
        
        X = df.select_dtypes(include=[np.number])
        
        X = X.drop(columns=ignore_cols, errors='ignore')
        
        bad_rows = X.isin([np.nan, np.inf , -np.inf]).any(axis=1)

        if (bad_rows.sum() / X.size) * 100 <= 5 :
            X = X[~bad_rows]
            df = df.loc[X.index]
        else :
            X = X.replace([np.inf, -np.inf, np.nan],0)

        y_raw = df['Binary_Label']
        
        # Skip if file is empty or has no features
        if X.shape[1] == 0 or len(y_raw) == 0:
            continue

        # Encode Labels (OK=0, KO=1)
        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        if algorithm == "rf" :
            # --- OPTION B: RANDOM FOREST ---
            # 1. Measure Sensor Quality (Model Accuracy)
            # We use a small Random Forest to see if this sensor is "Smart" or "Dumb"
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            
            # 2. Extract Feature Importance
            model.fit(X, y) # Retrain on full data
            importances = model.feature_importances_

        elif algorithm == 'dt':
            # --- OPTION B: DECISION TREE ---
            # No Scaling Needed
            X_final = X
            
            # Simple single tree
            model = DecisionTreeClassifier(random_state=42)

            model.fit(X_final, y)
            
            # Use Gini Importance (Same metric as RF)
            importances = model.feature_importances_

        elif algorithm == "lr" :
            # --- OPTION C: LOGISTIC REGRESSION ---
            # 1. Scaling is MANDATORY for LR
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_final = pd.DataFrame(X_scaled, columns=X.columns)
            
            # 2. Define Model
            model = LogisticRegression(max_iter=1000, solver='lbfgs')
            
            # 3. Train
            model.fit(X_final, y)
            
            # 4. Get Importance (Coefficients)
            # Take absolute value because -2.5 is just as important as +2.5
            importances = np.abs(model.coef_[0])
            
            # Normalize to 0-1 for fair plotting comparison
            if np.sum(importances) > 0:
                importances = importances / np.sum(importances)

        # Cross-validation gives a reliable accuracy score (e.g., 0.98 or 0.55)
        # cv=3 is fast enough for analysis
        acc_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        model_acc = acc_scores.mean()
        
        # Find the single best feature for this sensor (for the print summary)
        # best_idx = importances.argmax()
        # best_feat_name = X.columns[best_idx]
        
        # print(f"{sensor_name:<30} | {model_acc:.1%}    | {best_feat_name}")

        # 4. Save Details for Global Plotting
        for name, score in zip(X.columns, importances):
            global_ranking.append({
                'Sensor': sensor_name,
                'Feature': name,
                'Local_Importance': score,
                'Sensor_Accuracy': model_acc,
                # WEIGHTED SCORE: Feature Quality * Sensor Quality
                'Global_Score': score * model_acc 
            })

    # ==========================================
    # VISUALIZATION
    # ==========================================
    if not global_ranking: return

    results_df = pd.DataFrame(global_ranking)
    
    # Sort by Global Score to find the "Champions"
    results_df = results_df.sort_values(by='Global_Score', ascending=False)
    
    # PLOT 1: Who is the Best Sensor?
    # We take the max accuracy per sensor
    sensor_scores = results_df.groupby('Sensor')['Sensor_Accuracy'].max().sort_values(ascending=False)
    
    fig1, ax1 = plt.subplots(1,1,figsize=(10, 6))
    sns.barplot(x=sensor_scores.values, y=sensor_scores.index, palette='magma',ax = ax1, hue = sensor_scores.index, legend= False)
    ax1.set_title("Sensor Reliability Tournament (Accuracy)")
    ax1.set_xlabel("Classification Accuracy (1.0 = Perfect)")
    ax1.axvline(0.5, color='red', linestyle='--', label='Random Guessing')
    fig1.tight_layout()
    
    
    # PLOT 2: The Top 20 Features Overall
    fig2, ax2 = plt.subplots(1,1,figsize=(12, 8))
    num_top_features = min(len(results_df),20)
    top_features = results_df.head(num_top_features)
    sns.barplot(x='Global_Score', y='Feature', hue='Sensor', data=top_features, dodge=False, ax = ax2)
    ax2.set_title(f"Top {num_top_features} Most Important Features (Weighted by Accuracy)")
    ax2.set_xlabel("Global Importance Score")
    fig2.tight_layout()
    
    plt.close(fig1)
    plt.close(fig2)
    
    return (True, (sensor_scores, top_features, ax1, ax2))

def generate_summary_string(sensor_scores, top_features):
    """
    Converts the analysis results into a text prompt for the LLM.
    """
    summary = "### AUTOMATED ANALYSIS REPORT ###\n\n"
    
    # 1. Summarize Sensor Reliability
    summary += "--- SENSOR RELIABILITY (Model Accuracy) ---\n"
    # Convert Series to string table
    summary += sensor_scores.to_markdown() 
    summary += "\n\n"
    
    # 2. Summarize Top Features
    summary += "--- TOP PREDICTIVE FEATURES (Global Weighted Score) ---\n"
    # Select only relevant columns for the LLM to read
    cols_to_show = ['Sensor', 'Feature', 'Sensor_Accuracy', 'Global_Score']
    # Limit to top 5 or 10 to save tokens
    summary += top_features[cols_to_show].head(5).to_markdown(index=False)
    
    return summary