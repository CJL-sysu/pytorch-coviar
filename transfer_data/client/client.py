# client.py
# ! /usr/bin/env python
# -*- coding: utf-8 -*-
# from __future__ import print_function

import os
import socket
import time
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import numpy as np

# config begin

# ip address of server
ip = "8.130.16.177"
# server should listen on this port
port = 6001
listen_dir = "video"
tmp_dir = "tmp"
send_dir = "send"

# config end


def send_file(server_ip, filename):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_ip, port))
        file_path = send_dir + "/" + filename
        client_socket.send(filename.encode())
        client_socket.close()
        #
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_ip, port))
        with open(file_path, "rb") as file:
            while True:
                data = file.read(1024)
                if not data:
                    break
                client_socket.send(data)

        print(f"Sent {filename} to server.")
        client_socket.close()
        os.system(f"rm {file_path}")
    except Exception as e:
        print(f"Error sending {filename}: {e}")


class FileMonitorHandler(FileSystemEventHandler):
    def __init__(self, **kwargs):
        super(FileMonitorHandler, self).__init__(**kwargs)
        # 监控目录
        self._watch_path = listen_dir

    # 重写文件创建函数，文件改变都会触发文件夹变化
    def on_closed(self, event):
        if not event.is_directory:  # 文件创建都会触发文件夹变化
            file_path = event.src_path  # file_path是相对路径，例如video/hello.264
            print("文件修改完成 %s " % file_path)
            os.system(f"mv {file_path} ./tmp")
            filename = file_path[
                (len(listen_dir) + 1) :
            ]  # 只保留文件名而去掉路径，例如hello.264
            os.system(
                f"cd tmp && ffmpeg -i {filename} -c:v libx264 -profile:v baseline -g 10 -sc_threshold 0 -crf 28 -an -refs 10 -preset veryslow -direct-pred auto standard_{filename}"
            )
            mp4_filename = (
                "standard_" + filename[0 : len(filename) - 3] + "mp4"
            )  # .264 -> .mp4 ,转化后的文件名，例如standard_hello.mp4
            os.system(
                f"cd tmp && ffmpeg -i standard_{filename} -c:v copy {mp4_filename}"
            )
            os.system(
                f"cd tmp && ffmpeg -i {mp4_filename} -c:v mpeg4 -f rawvideo mpeg4_{mp4_filename}"
            )
            # a = load('mpeg4_{mp4_filename}', 3, 8, 1, True)
            npy_filename = (
                filename[0 : len(filename) - 3] + "npy"
            )  # 保存numpy矩阵文件，例如hello.npy
            # np.save(npy_filename, a)
            os.system(f"python load.py {tmp_dir} mpeg4_{mp4_filename} {npy_filename}")
            os.system(f"cd tmp && mv {npy_filename} ../send")
            os.system(f"rm tmp/*")
            send_file(ip, npy_filename)


if __name__ == "__main__":
    if not os.path.exists(listen_dir):
        os.makedirs(listen_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    if not os.path.exists(send_dir):
        os.makedirs(send_dir)
    event_handler = FileMonitorHandler()
    observer = Observer()
    observer.schedule(event_handler, path=listen_dir, recursive=True)  # recursive递归的
    observer.start()
    observer.join()
