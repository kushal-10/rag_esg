import pandas as pd
import os 
import json

def generate_targets(mapping_path: str = os.path.join("data", "maps.csv"), save_path: str = os.path.join("data", "targets.json")):
    """
    Load the CSV data from Refinitiv Internal Database and get the sub-targets for each sustainable development goal 
    Args:
        mapping_path: Path of csv file containing the Refinitiv definitions of SDGs
        save_path: File name to save the sub-targets for each goal
    """

    # Load the mapping CSV
    mapping_df = pd.read_csv(mapping_path) 

    # Counter for SDGoals
    GOAL = 0
    targets = {}

    prev_val = mapping_df.iloc[0]['Field Description']
    for i in range(1, len(mapping_df)):
        curr_val = mapping_df.iloc[i]['Field Description']

        # Skip NAN, Only consider string vals
        if type(curr_val) == str and type(prev_val) == str:
            # Continued Sub-targets
            targets[str(GOAL)].append(curr_val)
        elif type(curr_val) == str and type(prev_val) != str:
            # Change from nan to str
            # Update Goal
            GOAL += 1
            targets[str(GOAL)] = [curr_val]
        prev_val = curr_val
    
    with open(save_path, 'w') as f:
        json.dump(targets, f, indent=4)
    print(f"Saved Sub-Target definitions to : {save_path}")


if __name__ == '__main__':
    generate_targets()