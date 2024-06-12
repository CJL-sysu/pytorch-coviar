
import os
import socket
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from loader import get_frames_file



def send_file(args, filename):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((args.ip, args.port))
        file_path = args.send_dir + "/" + filename
        client_socket.send(filename.encode())
        client_socket.close()
        #
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((args.ip, args.port))
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
    def __init__(self, args):
        super(FileMonitorHandler, self).__init__()
        # 监控目录
        self._watch_path = args.listen_dir
        self.args = args

    # 重写文件创建函数，文件改变都会触发文件夹变化
    def on_closed(self, event):
        if not event.is_directory:  # 文件创建都会触发文件夹变化
            file_path = event.src_path  # file_path是相对路径，例如video/hello.264
            print("文件修改完成 %s " % file_path)
            os.system(f"mv {file_path} ./tmp")
            filename = file_path[
                (len(self.args.listen_dir) + 1) :
            ]  # 只保留文件名而去掉路径，例如hello.264
            std_filename = "std_" + filename # 标准化，修复打乱的帧后的文件名，例如std_hello.264
            os.system(
                f"cd tmp && ffmpeg -i {filename} -c:v libx264 -profile:v baseline -g 10 -sc_threshold 0 -crf 28 -an -refs 10 -preset veryslow -direct-pred auto {std_filename}"
            )
            mp4_filename = (
                filename[0 : len(filename) - 3] + "mp4"
            )  # .264 -> .mp4 ,转化后的文件名，例如hello.mp4
            os.system(
                f"cd tmp && ffmpeg -i {std_filename} -c:v copy {mp4_filename}"
            )
            mpeg4_filename = ("pg4_" + mp4_filename) # 转化为mpeg4格式之后的文件名，例如pg4_hello.mp4
            os.system(
                f"cd tmp && ffmpeg -i {mp4_filename} -c:v mpeg4 -f rawvideo {mpeg4_filename}"
            )
            bin_filename = mp4_filename[0: len(mp4_filename)-3] + "bin" # 提取的特征矩阵列表文件名，例如hello.bin
            #get_frames_file(mpeg4_filename, self.args, bin_filename)
            os.system(
                f"python loader.py --video_path {mpeg4_filename} --representation {self.args.representation} --test_segments {self.args.test_segments} --store_file {bin_filename}" + "--no_accumulation" if self.args.no_accumulation else ""
            )
            
            os.system(f"cd tmp && mv {bin_filename} ../send")
            os.system(f"rm tmp/*")
            send_file(self.args, bin_filename)


def transfer_worker(args):
    event_handler = FileMonitorHandler(args)
    observer = Observer()
    observer.schedule(event_handler, path=args.listen_dir, recursive=True)  # recursive递归的
    observer.start()
    observer.join()
