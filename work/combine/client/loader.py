import argparse
from coviar import get_num_frames
from coviar import load
import numpy as np
import pickle


GOP_SIZE = 12


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
            #     img = np.zeros((256, 256, 2)) if self._representation == 'mv' else np.zeros((256, 256, 3))
            # else:
            #     if self._representation == 'mv':
            #         img = clip_and_scale(img, 20)
            #         img += 128
            #         img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
            #     elif self._representation == 'residual':
            #         img += 128
            #         img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

            # if self._representation == 'iframe':
            #     img = color_aug(img)

            #     # BGR to RGB. (PyTorch uses RGB according to doc.)
            #     img = img[..., ::-1]

            frames.append(img)
        
        return frames

def save_list_to_bin_file(list_, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(list_, f)

def load_list_from_bin_file(file_path):
    with open(file_path, 'rb') as f:
        list_ = pickle.load(f)
    return list_

def get_frames_file(file_name, args, bin_filename):
    data = CoviarData(file_name, args.representation, args.test_segments, not args.no_accumulation)
    frames = data.get_mat()
    save_list_to_bin_file(frames, bin_filename)
    

# !deprecated
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