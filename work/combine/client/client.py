import os
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import numpy as np
import argparse
from transfer import transfer_worker

def confirm_not_None(args, name):
    if getattr(args, name) is None:
        raise ValueError(f"{name} must be specified")

def mkdir(args):
    if not os.path.exists(args.listen_dir):
        os.makedirs(args.listen_dir)
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)
    if not os.path.exists(args.send_dir):
        os.makedirs(args.send_dir)

def parse_args():
    # parse args
    parser = argparse.ArgumentParser(
    description="load video for coviar")
    #parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--representation', type=str, choices=['iframe', 'residual', 'mv'])
    parser.add_argument('--test_segments', type=int, default=25)
    parser.add_argument('--no_accumulation', action='store_true',
                        help='disable accumulation of motion vectors and residuals.')
    #parser.add_argument('--store_file', type=str, default= "frames.bin")
    
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6001)
    parser.add_argument('--listen_dir', type=str, default='video')
    parser.add_argument('--tmp_dir', type=str, default='tmp')
    parser.add_argument('--send_dir', type=str, default='send')

    args = parser.parse_args()
    #confirm_not_None(args, 'video_path')
    confirm_not_None(args, 'representation')
    #confirm_not_None(args, 'ip')
    
    return args

def main():
    args = parse_args()
    mkdir(args)
    transfer_worker(args)