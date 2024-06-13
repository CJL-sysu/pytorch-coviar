## client端

部署在本地，用于从格式化为mpeg4的视频文件提取特征矩阵列表



### 启动client

在test文件夹中，提供一个 mpeg4 格式的视频供测试。实际使用中，这个视频来自摄像头实时拉取的视频切片，且经过 mpeg4 格式转换

```bash
python client.py --video_path ${your-video.mp4} --representation mv
# 例如
python client.py --video_path test/April_09_brush_hair_u_nm_np1_ba_goo_0.mp4 --representation mv
```

启动之后，如果不使用 `--store_file` 指定文件名，默认会保存一个文件 `frames.bin` , 它是二进制保存的特征矩阵列表

`frames.bin` 加载到 python 中是一个 list ，其中元素为 numpy 矩阵

1. （检验coviar的例子中的大小为 681K， shape(256, 340, 2)）长度25，总大小 6.3M

2. 选用mv，大小4.2M