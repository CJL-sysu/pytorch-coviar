import argparse
from coviar import get_num_frames
from coviar import load
import numpy as np
import random
import cv2
import pickle

def color_aug(img, random_h=36, random_l=50, random_s=50):
    ''' 
    此函数对输入图像执行颜色增强:
    将图像从BGR颜色空间转换为HLS颜色空间。
    随机调整色调h、亮度l和饱和度s,
    将图像转换回BGR颜色空间并返回
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(float)
    #使用 OpenCV 的 cvtColor 函数将图像从 BGR 颜色空间转换为 HLS 颜色空间，并将图像数组转换为浮点型（float），以便后续处理。

    h = (random.random() * 2 - 1.0) * random_h
    l = (random.random() * 2 - 1.0) * random_l
    s = (random.random() * 2 - 1.0) * random_s
    #生成三个随机值 h、l 和 s。random.random() 生成一个在 [0, 1) 之间的随机数，通过 (random.random() * 2 - 1.0) 将其映射到 [-1, 1) 区间，再乘以对应的 random_h、random_l 和 random_s，得到色调、亮度和饱和度的随机变化量。


    img[..., 0] += h
    img[..., 0] = np.minimum(img[..., 0], 180)
    #将生成的随机值 h 加到图像的色调通道（H），并使用 np.minimum 函数确保色调值不超过 180（HLS 色调的最大值）。

    img[..., 1] += l
    img[..., 1] = np.minimum(img[..., 1], 255)
    #将生成的随机值 l 加到图像的亮度通道（L），并使用 np.minimum 函数确保亮度值不超过 255（HLS 亮度的最大值）。

    img[..., 2] += s
    img[..., 2] = np.minimum(img[..., 2], 255)
    #将生成的随机值 s 加到图像的饱和度通道（S），并使用 np.minimum 函数确保饱和度值不超过 255（HLS 饱和度的最大值）。

    img = np.maximum(img, 0)
    #使用 np.maximum 函数确保图像的各个通道值不小于 0。
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HLS2BGR)
    #将图像转换回无符号 8 位整型（uint8），然后使用cvtColor 函数将图像从 HLS 颜色空间转换回 BGR 颜色空间，并返回增强后的图像。
    return img

GOP_SIZE = 12

def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)

def get_gop_pos(frame_idx, representation):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos

class CoviarData:
    def __init__(self, video_path, representation, test_segments, accumulate):
        self._video_path = video_path
        self._representation = representation
        self._num_segments = test_segments
        self._num_frames = get_num_frames(video_path)
        self._accumulate = accumulate

    def _get_test_frame_index(self, num_frames, seg):
        if self._representation in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if self._representation in ['mv', 'residual']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, self._representation)
    
    def get_mat(self):
        if self._representation == 'mv':
            representation_idx = 1
        elif self._representation == 'residual':
            representation_idx = 2
        else:
            representation_idx = 0
        frames = []
        for seg in range(self._num_segments): # num_segments为加载切片的数量
            gop_index, gop_pos = self._get_test_frame_index(self._num_frames, seg)
            img = load(self._video_path, gop_index, gop_pos,
                       representation_idx, self._accumulate)
            if img is None:
                print('Error: loading video %s failed.' % self._video_path)
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
        
        return frames

def save_list_to_bin_file(list_, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(list_, f)

def load_list_from_bin_file(file_path):
    with open(file_path, 'rb') as f:
        list_ = pickle.load(f)
    return list_

def parse_args():
    # parse args
    parser = argparse.ArgumentParser(
    description="load video for coviar")
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--representation', type=str, choices=['iframe', 'residual', 'mv'])
    parser.add_argument('--test_segments', type=int, default=25)
    parser.add_argument('--no_accumulation', action='store_true',
                        help='disable accumulation of motion vectors and residuals.')
    parser.add_argument('--store_file', type=str, default= "frames.bin")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.representation is None:
        raise ValueError('Representation not specified.')
    data = CoviarData(args.video_path, args.representation, args.test_segments, not args.no_accumulation)
    frames = data.get_mat()
    #print(frames)
    save_list_to_bin_file(frames, args.store_file)
    
if __name__ == '__main__':
    main()