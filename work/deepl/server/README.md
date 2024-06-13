## server端深度学习实现

本文件夹下还包含预训练的模型参数(*.pth.tar文件)

在test文件夹中，提供了可供测试的视频特征提取矩阵，只支持mv，在实际应用中，这类文件应由本地上传。

> 注意：test_segments和representation必须和client端保持一致

```bash
# 使用mv
python server.py --gpus 0 --arch resnet18 --data_name hmdb51 --representation mv --weights hmdb51_mv_model_mv_model_best.pth.tar --file_path test/frames.bin
# 使用residual
python server.py --gpus 0 --arch resnet18 --data_name hmdb51 --representation residual --weights hmdb51_residual_model_residual_model_best.pth.tar --file_path test/frames.bin
```

示例
```bash
$ python server.py --gpus 0 --arch resnet18 --data_name hmdb51 --representation mv --weights hmdb51_mv_model_mv_model_best.pth.tar --file_path test/frames.bin

Initializing model:
    base model:         resnet18.
    input_representation:     mv.
    num_class:          51.
    num_segments:       25.
        
/root/miniconda3/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/root/miniconda3/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
model epoch 236 best prec@1: 30.130720138549805
server.py:137: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  input_var = torch.autograd.Variable(data, volatile=True)
the classify result is 33#brush_hair
```
结果为梳头，识别是准确的
经过小范围测试，选用mv和residual都能准确识别动作类型，iframe因为可以逆向恢复出完整图片，不具备隐私保护的能力，没有测试过