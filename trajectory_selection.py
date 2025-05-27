import numpy as np

def load_trajectories_data(file_path):
    loaded = np.load(file_path, allow_pickle=True)
    all = []
    for (start, end), lang in zip(loaded.all()["info"]["indx"], loaded.all()["language"]["ann"]):
        all.append([start, end, lang])
    return all
datasets = load_trajectories_data("/standard/liverobotics/calvin_task_D_D/training/lang_annotations/auto_lang_ann.npy")

print(datasets)
# [315660, 315724, 'move the door to the left side']