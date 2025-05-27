import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import random
from tqdm import tqdm
        
class subgoalDataset(Dataset):
    def __init__(self, keyframes_file_path, data_dir, token_transform,debug_data_num , feature_dir = None):
        self.loaded_look_up = np.load(keyframes_file_path, allow_pickle=True)
        np.random.shuffle(self.loaded_look_up)
        if(debug_data_num != -1):
            self.loaded_look_up = self.loaded_look_up[0:debug_data_num]
        self.data_dir = data_dir
        self.token_transform = token_transform
        self.feature_dir = feature_dir
        self.start_list = []
        self.end_list = []
        self.keyframes_list = []
        self.task_list = []
        self.lang_list = []
        
        self.pil_dict = {}
        
        for start, end, keyframes, task, lang in tqdm(self.loaded_look_up):
            start_file_name = "episode_" + format(int(start), '07d') + ".npz"
            end_file_name = "episode_" + format(int(end), '07d') + ".npz"
            start_file_path = os.path.join(self.data_dir, start_file_name)
            end_file_path = os.path.join(self.data_dir, end_file_name)

            start_npz = np.load(start_file_path, allow_pickle = True)
            end_npz = np.load(end_file_path, allow_pickle = True)
            try:
                start_rgb_static = start_npz["rgb_static"]
                end_rgb_static = end_npz["rgb_static"]
            except:
                start_rgb_static = start_npz["rgb_low_static"]
                end_rgb_static = end_npz["rgb_low_static"]
            start_pil = Image.fromarray(start_rgb_static)
            end_pil = Image.fromarray(end_rgb_static)

            lang_token = self.token_transform(lang).squeeze()

            id_list = [start]
            for k in keyframes:
                id_list.append(k)
            if(feature_dir is None):
                all_image_features = None
            else:
                all_image_features = self.load_features(id_list)
            # self.keyframes_features_list.append(all_image_features)
            
            self.pil_dict[str(start)] = start_pil
            self.pil_dict[str(end)] = end_pil
            
            self.start_list.append(start)
            self.end_list.append(end)
            self.keyframes_list.append(all_image_features)
            # self.task_list = []
            self.lang_list.append(lang_token)
            

    def load_features(self, id_list):
        feature_list = []
        for id in id_list:
            id = format(int(id), '07d')
            name = f"episode_{id}_clip_feature.npy"
            feature = np.load(os.path.join(self.feature_dir, name))
            feature_list.append(feature)
        return feature_list    
        
        
    def file_name_format(self, id):
        id = format(int(id), '07d')
        return "episode_" + id + r".npz"
    
    def __len__(self):
        return len(self.loaded_look_up)

    def __getitem__(self, idx):
        start_id = self.start_list[idx]
        end_id = self.end_list[idx]
        all_image_features = self.keyframes_list[idx]
        lang_token = self.lang_list[idx]
        
        start_pil = self.pil_dict[str(start_id)]
        end_pil = self.pil_dict[str(end_id)]
        
        return {
            "start_image": start_pil,
            "end_image": end_pil,
            "lang": lang_token,
            "keyframes_feature" : all_image_features
        }
        
if __name__ == "__main__":
    print("main")