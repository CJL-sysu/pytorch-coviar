### client端

启动client

```bash
python work/client/client.py --video_path ${your-video.mp4}
# 例如
python work/client/client.py --video_path April_09_brush_hair_u_nm_np1_ba_goo_0.mp4
```

启动之后，会保存一个文件frames.bin

frams.bin 是一个list，其中元素为numpy矩阵（检验coviar的例子中的大小为 681K， shape(256, 340, 2)）

长度25，总大小 6.3M