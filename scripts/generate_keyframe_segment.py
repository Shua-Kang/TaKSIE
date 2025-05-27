import numpy as np
import os
from tqdm import tqdm
import argparse

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Process keyframe data and create structured trajectory segments.")
    parser.add_argument("--input_npy_path", type=str, help="Path to the input .npy file (e.g., keyframes_full_info_ABC.npy).")
    parser.add_argument("--output_npy_path", type=str, help="Full path for the output .npy file (e.g., /path/to/output/trajectory_segments.npy).")
    args = parser.parse_args()

    # --- Load Data ---
    print(f"Loading data from: {args.input_npy_path}")
    try:
        loaded_keyframe_info = np.load(args.input_npy_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_npy_path}")
        return
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # --- Process Data ---
    trajectory_segments = [] # 
    print("Processing keyframes into trajectory segments...")

    for item in tqdm(loaded_keyframe_info):
        
        start_ts, end_ts, lang_instruction, keyframe_timestamps, task_identifier, _ = item[0], item[1], item[2], item[3], item[4], item[5]
        # (task_id was item[5], renamed to _ if not used further in this specific processing logic)

        current_full_sequence = [start_ts]
        if isinstance(keyframe_timestamps, (list, np.ndarray)):
            for individual_kf_ts in keyframe_timestamps:
                current_full_sequence.append(individual_kf_ts)
        else:
            # Assuming if not a list/array, it might be a single keyframe or needs specific handling
            current_full_sequence.append(keyframe_timestamps) # Add it directly
        current_full_sequence.append(end_ts)

        for i in range(len(current_full_sequence) - 1): # Iterate up to the second to last element
            current_state_kf = current_full_sequence[i]
            next_state_kf = current_full_sequence[i+1]
            history_subgoals_kfs = current_full_sequence[0:i+1] # Keyframes from start up to and including current
            
            trajectory_segments.append([ 
                current_state_kf,
                next_state_kf,
                history_subgoals_kfs,
                task_identifier,
                lang_instruction
            ])

    # --- Save Data ---
    trajectory_segments_array = np.array(trajectory_segments, dtype=object) 
    print(f"Saving processed trajectory segments to: {args.output_npy_path}")
    try:
        output_dir = os.path.dirname(args.output_npy_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
            
        np.save(args.output_npy_path, trajectory_segments_array, allow_pickle=True) 
        print("Processed trajectory segments saved successfully.")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    main()