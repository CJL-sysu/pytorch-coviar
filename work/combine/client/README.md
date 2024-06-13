## client端

部署在本地:

1. 自动将视频进行格式转换，调用loader.py提取特征矩阵列表，然后上传至服务器

2. loader.py可单独运行，用于从格式化为mpeg4的视频文件提取特征矩阵列表

### 环境配置

需要在client端配置 coviar 环境，详见 https://docs.qq.com/aio/DVGFqeGdWS3NhclZK?p=iC8mFVF7uD4XJqzRMiXdmx ，提供x86和arm两种架构的配置教程

### 启动client

启动视频推流


```bash
nohup rpicam-vid -t 0 --inline -o - | cvlc stream:///dev/stdin --sout '#rtp{sdp=rtsp://:4336/stream1}' :demux=h264 2>&1 &
```

启动client.py

```bash
/bin/python client.py --representation mv --ip ${server-ip} --port ${server-port}
```

启动视频分段拉流
```bash
ffmpeg -i rtsp://127.0.0.1:4336/stream1 -c:v copy -f segment -segment_time 10 -reset_timestamps 1 "video/output_%03d.264"
```


### 测试loader.py
在test文件夹中，提供一个 mpeg4 格式的视频供测试。实际使用中，这个视频来自摄像头实时拉取的视频切片，且经过 mpeg4 格式转换


```bash
python loader.py --video_path ${your-video.mp4} --representation mv
# 例如
python loader.py --video_path test/April_09_brush_hair_u_nm_np1_ba_goo_0.mp4 --representation mv --store_file frames_mv.bin
python loader.py --video_path test/April_09_brush_hair_u_nm_np1_ba_goo_0.mp4 --representation residual --store_file frames_res.bin
```

pyenv 要想支持 watchdog 需要安装的时候就注意...