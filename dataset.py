"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors, 
or residuals) for training or testing.
"""

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data

from coviar import get_num_frames
from coviar import load
from transforms import color_aug


GOP_SIZE = 12


def clip_and_scale(img, size):
    """
    这个函数对输入的图像进行裁剪和缩放。
    img: 输入的图像数组。
    size: 用于缩放的尺寸参数。
    ---
    img * (127.5 / size): 首先，将图像数组中的每个像素值乘以一个缩放因子。缩放因子是 127.5 / size，这是一个归一化过程。
    .astype(np.int32): 将缩放后的图像数组转换为 np.int32 类型，这通常是为了确保像素值是整数。
    """
    return (img * (127.5 / size)).astype(np.int32)


def get_seg_range(n, num_segments, seg, representation):
    """
    这个函数计算视频段（segment）的起始和结束帧的索引。

    参数
    n: 总帧数。
    num_segments: 需要划分的段数。
    seg: 当前段的索引。
    representation: 表示类型，可以是 'residual' 或 'mv' 等。
    """
    if representation in ['residual', 'mv']:
        n -= 1

    seg_size = float(n - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg+1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['residual', 'mv']:
        # Exclude the 0-th frame, because it's an I-frame.
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end


def get_gop_pos(frame_idx, representation):
    """
    用于计算给定帧在视频的GOP（Group of Pictures，图像组）中的位置。
    """
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos


class CoviarDataSet(data.Dataset):
    def __init__(self, data_root, # 数据集路径
                 data_name, # 数据集名称
                 video_list, # 数据集标注文件，在/data/datalists
                 representation, # 表示类型，可以是 'frame', 'mv', 'residual' 
                 transform, # 对视频操作
                 num_segments, # 默认25
                 is_train, # 是否训练
                 accumulate): # 默认可累计

        self._data_root = data_root
        self._data_name = data_name
        self._num_segments = num_segments
        self._representation = representation
        self._transform = transform
        self._is_train = is_train
        self._accumulate = accumulate

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_list(video_list)

    def _load_list(self, video_list):
        self._video_list = []
        with open(video_list, 'r') as f: # 读取数据集标注文件
            for line in f:
                video, _, label = line.strip().split()
                video_path = os.path.join(self._data_root, video[:-4] + '.mp4')# 找到标签对应视频的真实路径
                self._video_list.append((
                    video_path,
                    int(label),
                    get_num_frames(video_path))) # counting the number of frames in a video

        print('%d videos loaded.' % len(self._video_list))

    def _get_train_frame_index(self, num_frames, seg):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
                                                 representation=self._representation)

        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return get_gop_pos(v_frame_idx, self._representation)

    def _get_test_frame_index(self, num_frames, seg):
        if self._representation in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if self._representation in ['mv', 'residual']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, self._representation)

    def __getitem__(self, index):

        if self._representation == 'mv':
            representation_idx = 1
        elif self._representation == 'residual':
            representation_idx = 2
        else:
            representation_idx = 0


        if self._is_train:
            video_path, label, num_frames = random.choice(self._video_list)
        else:
            video_path, label, num_frames = self._video_list[index]

        frames = []
        for seg in range(self._num_segments): # num_segments为加载切片的数量

            if self._is_train:
                gop_index, gop_pos = self._get_train_frame_index(num_frames, seg)
            else:
                gop_index, gop_pos = self._get_test_frame_index(num_frames, seg)

            img = load(video_path, gop_index, gop_pos,
                       representation_idx, self._accumulate)
            # coviar 的 load 函数，后续实现中，load在本地树莓派完成，只需要将img换成从本地上传的 numpy 矩阵

            if img is None:
                print('Error: loading video %s failed.' % video_path)
                img = np.zeros((256, 256, 2)) if self._representation == 'mv' else np.zeros((256, 256, 3))
            else:
                if self._representation == 'mv':
                    img = clip_and_scale(img, 20)
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
                elif self._representation == 'residual':
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

            if self._representation == 'iframe':
                img = color_aug(img)

                # BGR to RGB. (PyTorch uses RGB according to doc.)
                img = img[..., ::-1]

            frames.append(img)
        
        # 此处截断，为clinet和server分界

        frames = self._transform(frames)

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        input = torch.from_numpy(frames).float() / 255.0

        if self._representation == 'iframe':
            input = (input - self._input_mean) / self._input_std
        elif self._representation == 'residual':
            input = (input - 0.5) / self._input_std
        elif self._representation == 'mv':
            input = (input - 0.5)

        return input, label

    def __len__(self):
        return len(self._video_list)
