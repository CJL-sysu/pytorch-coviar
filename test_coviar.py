from coviar import load
import numpy as np
from coviar import get_num_frames
from tqdm import tqdm
# a = load('Glory_shoot_gun_u_nm_np1_le_med_47.mp4', 3, 8, 1, True)
root_path = "data/hmdb51/mpeg4_videos"
# a = load(
#     "Glory_shoot_gun_u_nm_np1_le_med_47.mp4", 3, 8, 1, True
# )
# b = get_num_frames("data/hmdb51/mpeg4_videos/chew/#2_Gum_chew_h_nm_np1_fr_med_0.mp4")
import os
for category in tqdm(os.listdir(root_path)):
    for video in os.listdir(os.path.join(root_path, category)):
        path = os.path.join(root_path, category, video)
        # if path == 'data/hmdb51/mpeg4_videos/somersault/Dive_and_roll_compilation_(Parkour)_somersault_f_cm_np1_ba_bad_3.mp4':
        #     print('\x1b[31m!!!\x1b[0m')
        #     continue

        try:
            a = load(path, 3, 8, 1, True)
            b = get_num_frames(path)
            # print(a.shape, b)
        except:
            print(path, end=" ")
            print("\x1b[31m!!!\x1b[0m")
