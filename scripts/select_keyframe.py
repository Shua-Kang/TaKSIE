import statsmodels.api as sm
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from r3m import load_r3m
import json
import os
import pandas as pd
from tqdm import tqdm
import argparse

lowess = sm.nonparametric.lowess

# Function to get embedding from disk
def get_emb_from_disk(image_id, r3m_feature_dir):
    return np.load(os.path.join(r3m_feature_dir, f"episode_{image_id}_r3m_feature.npy")).squeeze(0)

# Function to load trajectories data
def load_trajectories_data(file_path):
    loaded = np.load(file_path, allow_pickle=True)
    all_data = []
    for (start, end), lang, task in zip(loaded.all()["info"]["indx"], loaded.all()["language"]["ann"], loaded.all()["language"]["task"]):
        all_data.append([start, end, lang, task])
    return all_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process keyframes from datasets")
    parser.add_argument('--r3m_feature_dir',type=str,  help='Path to the R3M features directory')
    parser.add_argument('--lang_annotations_path' ,type=str, help='Path to the language annotations file')
    parser.add_argument('--data_path' ,type=str,  help='Path to the data directory')
    parser.add_argument('--output_dir', type=str, help='Directory to save the output')
    parser.add_argument('--delta_1', default = -0.001,type=float, help='Directory to save the output')
    parser.add_argument('--delta_2', default = 0.001,type=float, help='Directory to save the output')

    args = parser.parse_args()

    r3m_feature_dir = args.r3m_feature_dir
    lang_annotations_path = args.lang_annotations_path
    data_path = args.data_path
    output_dir = args.output_dir
    output_file = os.path.join(output_dir, "selected_keyframes.npy")
    datasets = load_trajectories_data(lang_annotations_path)
    scale = 1
    threshold_2 = args.delta_2
    threshold = args.delta_1
    interval = 7
    keyframes = []

    for num, (start, end, lang, task) in enumerate(tqdm(datasets)):
        keyframe = []
        offset = 0
        while True:
            distance_list_before = []
            start = int(start)
            end = int(end)
            var_point = start + offset
            if var_point == end:
                keyframe.append(end)
                offset = end - start
                keyframes.append(keyframe)
                break
            if var_point > end:
                print("error3")
                exit(0)
            var_point_file_name = format(var_point, '07d')
            var_emb = torch.tensor(get_emb_from_disk(var_point_file_name, r3m_feature_dir))
            for id in range(var_point, end + 1):
                id = format(id, '07d')
                cur_emb = torch.tensor(get_emb_from_disk(id, r3m_feature_dir))
                distance_list_before.append(torch.norm((var_emb - cur_emb), p=2).item())
            x1 = range(0, end - var_point + 1)
            distance_list = lowess(distance_list_before, x1, frac=1./6.)[:, 1]
            distance_list = np.array(distance_list)
            distance_list = (distance_list - min(distance_list)) / (max(distance_list) - min(distance_list))
            x2 = range(0, end - var_point - scale + 1)
            intercept = [distance_list[i + scale] - distance_list[i] for i in range(len(distance_list) - scale)]
            
            for i, dx in enumerate(intercept):
                if (i + 1) < interval:
                    continue
                elif (i + 1) <= len(intercept) - interval and dx <= threshold and intercept[i-1] >= threshold_2:
                    offset = offset + i + 1
                    keyframe.append(start + offset)
                    break
                elif (i + 1) <= len(intercept) - interval and dx > threshold:
                    continue
                elif len(intercept) - interval <= 0:
                    keyframe.append(end)
                    offset = end - start
                    break
                elif (i + 1) == len(intercept) or (len(intercept) - (i + 1)) < interval:
                    keyframe.append(end)
                    offset = end - start
                    break
                else:
                    continue

        if keyframe[-1] != end:
            print("error1")
            exit(0)

    new_frames = []
    for num, (start, end, lang, task) in enumerate(datasets):
        new_frames.append([start, end, lang, keyframes[num][0:-1], task])

    new_frame_with_task_type = []

    for start, end, lang, kfs, task in new_frames:
        new_frame_with_task_type.append([start, end, lang, kfs, task, task])

    new_frame_with_task_type = np.array(new_frame_with_task_type, dtype=object)
    np.save((output_file), new_frame_with_task_type, allow_pickle=True)
