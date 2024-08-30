from coviar import load
import numpy as np
from coviar import get_num_frames

# a = load('data/hmdb51/mpeg4_videos/chew/Big_League_Chew_chew_h_nm_np1_fr_goo_0.mp4', 3, 8, 1, True)
a = load("../../data/charades/data/Charades_v1_480/mpeg4_videos/Drinking_from_a_cup(glass,_bottle)/9NL5G1.mp4", 3, 8, 1, True)
b = get_num_frames("../../data/charades/data/Charades_v1_480/mpeg4_videos/Drinking_from_a_cup(glass,_bottle)/9NL5G1.mp4")
print(a)
print(b)