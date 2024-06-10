"""Functions for data augmentation and related preprocessing."""

import random
import numpy as np
import cv2


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


class GroupCenterCrop(object):
    ''' 
    对一组图像进行中心裁剪。
    输入大小，并将组中的每个图像裁剪到该大小。
    中心裁剪:通过计算起始坐标来获得的，基于图像的尺寸。
    '''

    def __init__(self, size):
        self._size = size

    def __call__(self, img_group):
        #取第一张图像的形状（高度 h、宽度 w 和通道数 _），假设所有图像具有相同的尺寸。
        h, w, _ = img_group[0].shape
        hs = (h - self._size) // 2
        ws = (w - self._size) // 2
        #计算高度和宽度的起始坐标 hs 和 ws，以进行中心裁剪(从图像的中心位置向两边扩展 _size 一半的距离）。
        return [img[hs:hs+self._size, ws:ws+self._size] for img in img_group]


class GroupRandomHorizontalFlip(object):
    '''
    对一组图像执行随机水平翻转。
    如果随机值小于0.5，则图像会水平翻转。
    对于运动矢量MV图像（由is_mv=True指示），还会对像素值进行额外调整。
    '''
    def __init__(self, is_mv=False):
        self._is_mv = is_mv

    def __call__(self, img_group, is_mv=False):
        if random.random() < 0.5:
            #使用random生成一个 [0, 1) 之间的随机数。如果这个随机数小于0.5，则执行图像翻转操作。
            ret = [img[:, ::-1, :].astype(np.int32) for img in img_group]
            #img[:, ::-1, :]表示将图像的列顺序反转，从而实现水平翻转。将图像转换为 np.int32 类型以便后续处理。
            if self._is_mv:
                #如果实例变量 _is_mv 为 True，将翻转的图片再进行操作。
                for i in range(len(ret)):
                    ret[i] -= 128
                    #将图像的像素值减去128。
                    ret[i][..., 0] *= (-1)
                    #将图像的第一个通道（假设是运动矢量的水平分量）乘以 -1，反转其方向。
                    ret[i] += 128
                    #将图像的像素值加上128，恢复到原始范围。
            return ret
        else:
            return img_group


class GroupScale(object):
    '''
    将一组图像缩放到指定大小。
    使用双线性插值对组中的每个图像进行调整。
    大小可以不同。
    '''
    def __init__(self, size):
        self._size = (size, size)

    def __call__(self, img_group):
        if img_group[0].shape[2] == 3:
        #如果图像的通道数为3（RGB图像），对图像组中的每一张图像使用cv2.resize函数进行缩放，目标大小为 _size。
            return [cv2.resize(img, self._size, cv2.INTER_LINEAR) for img in img_group]
        elif img_group[0].shape[2] == 2:
        #如果图像的通道数为2（运动矢量图像），对图像组中的每一张图像使用resize_mv函数进行缩放，目标大小为 _size。
            return [resize_mv(img, self._size, cv2.INTER_LINEAR) for img in img_group]
        else:
        #如果图像的通道数既不是3也不是2，是不支持的图像类型。
            assert False


class GroupOverSample(object):
    '''
    通过裁剪和翻转对一组图像进行过采样。
    如果提供了scale_size，它首先使用GroupScale对图像进行缩放。
    通过应用不同的偏移量生成指定crop_size的多个裁剪。
    对于MV图像，还会对像素值进行额外调整
    '''
    def __init__(self, crop_size, scale_size=None, is_mv=False):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        #如果crop_size是整数类型的话，转换为一个元组 (crop_size, crop_size)。

        if scale_size is not None:
        #如果提供了 scale_size，则创建一个 GroupScale 实例用于缩放图像，否则将 scale_worker 设置为 None。
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self._is_mv = is_mv
        #将 is_mv 存储在实例变量 _is_mv 中，指示是否处理运动矢量图像。

    def __call__(self, img_group):

        if self.scale_worker is not None:
        #如果存在 scale_worker，对图像组进行缩放。
            img_group = self.scale_worker(img_group)

        image_w, image_h, _ = img_group[0].shape
        #获取第一张图像的宽度 image_w、高度 image_h 和通道数 _。
        crop_w, crop_h = self.crop_size
        #获取裁剪宽度 crop_w 和裁剪高度 crop_h。

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        #使用 GroupMultiScaleCrop.fill_fix_offset 生成裁剪的偏移量（offsets，计算固定的裁剪偏移量。
        oversample_group = list()
        #创建一个空列表 oversample_group 用于存储过采样的图像。

        for o_w, o_h in offsets:
            for img in img_group:

                crop = img[o_w:o_w+crop_w, o_h:o_h+crop_h]
                #根据偏移量和裁剪大小，提取裁剪区域 crop。
                oversample_group.append(crop)
                #将裁剪的图像添加到 oversample_group 列表中。

                flip_crop = crop[:, ::-1, :].astype(np.int32)
                #对裁剪后的图像进行水平翻转，生成 flip_crop。
                if self._is_mv:
                #如果是运动矢量图像
                    assert flip_crop.shape[2] == 2, flip_crop.shape
                    flip_crop -= 128
                    flip_crop[..., 0] *= (-1)
                    flip_crop += 128
                oversample_group.append(flip_crop)
                #将处理后的翻转图像添加到 oversample_group 列表中。

        return oversample_group

def resize_mv(img, shape, interpolation):
    '''
    此函数调整运动矢量（MV）图像的大小。
    它将调整后的通道（MV的两个通道）堆叠以创建最终调整大小的MV图像。
    '''
    return np.stack([cv2.resize(img[..., i], shape, interpolation)
                     for i in range(2)], axis=2)
    #列表生成式对运动矢量图像的每个通道（两个）进行调整大小操作。


class GroupMultiScaleCrop(object):
    '''
    对一组图像执行多尺度裁剪。
    基于输入的尺度和最大失真采样裁剪大小。
    根据指定的选项，裁剪可以居中或随机采样。
    '''

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=False, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]

    def __call__(self, img_group):

        im_size = img_group[0].shape
        #获取第一张图像的尺寸 im_size。

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        #调用 _sample_crop_size 获取裁剪宽度 crop_w、裁剪高度 crop_h 以及偏移量 offset_w 和 offset_h

        crop_img_group = [img[offset_w:offset_w + crop_w, offset_h:offset_h + crop_h] for img in img_group]
        #对图像组中的每张图像进行裁剪

        if crop_img_group[0].shape[2] == 3:
        #如果裁剪后的图像是 RGB 图像（通道数为3），则使用 cv2.resize 函数进行缩放。
            ret_img_group = [cv2.resize(img, (self.input_size[0], self.input_size[1]),
                                        cv2.INTER_LINEAR)
                             for img in crop_img_group]
        elif crop_img_group[0].shape[2] == 2:
        #如果裁剪后的图像是 MV 图像（通道数为2），则使用 resize_mv 函数进行缩放。
            ret_img_group = [resize_mv(img, (self.input_size[0], self.input_size[1]), cv2.INTER_LINEAR)
                             for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]
        #获取图像的宽度 image_w 和高度 image_h。

        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        #计算基础尺寸 base_size（最小边长），并基于 scales 生成可能的裁剪尺寸 crop_sizes
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]
        #根据 input_size 生成裁剪高度 crop_h 和裁剪宽度 crop_w。

        pairs = []
        
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))
                    #生成所有可能的裁剪尺寸对 pairs，确保高度和宽度之间的差异不超过 max_distort。

        crop_pair = random.choice(pairs)
        #随机选择一个裁剪尺寸对 crop_pair。
        if not self.fix_crop:
        #如果 fix_crop 为 False，则随机生成裁剪的偏移量 w_offset 和 h_offset。
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
        #调用 _sample_fix_offset 方法获取固定偏移量。
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        '''
        定义 _sample_fix_offset 方法，用于生成固定的裁剪偏移量。
        '''
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        #调用 fill_fix_offset 方法生成所有可能的偏移量 offsets。
        return random.choice(offsets)
        #随机选择一个偏移量并返回。

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        '''
        定义了 fill_fix_offset 静态方法，用于生成固定的裁剪偏移量。
        '''
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4
        #计算水平步长 w_step 和垂直步长 h_step。

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center
        #生成基础的五个偏移量：左上、右上、左下、右下和中心。

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
        #more_fix_crop 为 True,则生成更多的偏移量，包括中心左右、上下、四分之一位置的偏移量。

        return ret
