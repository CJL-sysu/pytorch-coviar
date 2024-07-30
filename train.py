"""Run training."""

import shutil
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision

from dataset import CoviarDataSet
from model import Model
from train_options import parser
from transforms import GroupCenterCrop
from transforms import GroupScale

SAVE_FREQ = 40  # 保存频率
PRINT_FREQ = 20  # 打印频率
best_prec1 = 0  # 最佳精度

def main():
    global args
    global best_prec1
    args = parser.parse_args()  # 解析命令行参数

    print('Training arguments:')
    for k, v in vars(args).items():  # 打印所有训练参数
        print('\t{}: {}'.format(k, v))

    if args.data_name == 'ucf101':  # 数据集
        num_class = 101
    elif args.data_name == 'hmdb51':
        num_class = 51
    else:
        raise ValueError('Unknown dataset ' + args.data_name)  # 未知数据集

    model = Model(num_class, args.num_segments, args.representation,
                  base_model=args.arch)  # 创建模型
    # 加载训练好的模型参数
    if args.weights is not None:
        checkpoint = torch.load(args.weights) 
        print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

        base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
        model.load_state_dict(base_dict) # 导入模型参数
    else:
        print('No pretrained model')
        return
    
    print(model)

    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.train_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=model.get_augmentation(),
            is_train=True,
            accumulate=(not args.no_accumulation),
            ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)  # 创建训练数据加载器

    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.test_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=torchvision.transforms.Compose([
                GroupScale(int(model.scale_size)),
                GroupCenterCrop(model.crop_size),
                ]),
            is_train=False,
            accumulate=(not args.no_accumulation),
            ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)  # 创建验证数据加载器

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()  # 使用多个GPU
    cudnn.benchmark = True  # 启用cudnn自动调节

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key else 1.0

        if ('module.base_model.conv1' in key
                or 'module.base_model.bn1' in key
                or 'data_bn' in key) and args.representation in ['mv', 'residual']:
            lr_mult = 0.1
        elif '.fc.' in key:
            lr_mult = 1.0
        else:
            lr_mult = 0.01

        params += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]  # 设置参数

    optimizer = torch.optim.Adam(
        params,
        weight_decay=args.weight_decay,
        eps=0.001)  # 创建优化器
    criterion = torch.nn.CrossEntropyLoss().cuda()  # 定义损失函数

    for epoch in range(args.epochs):
        cur_lr = adjust_learning_rate(optimizer, epoch, args.lr_steps, args.lr_decay)  # 调整学习率

        train(train_loader, model, criterion, optimizer, epoch, cur_lr)  # 训练模型

        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion)  # 验证模型

            is_best = prec1 > best_prec1  # 检查是否为最佳模型
            best_prec1 = max(prec1, best_prec1)  # 更新最佳精度
            if is_best or epoch % SAVE_FREQ == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                    },
                    is_best,
                    filename='checkpoint.pth.tar')  # 保存模型检查点

def train(train_loader, model, criterion, optimizer, epoch, cur_lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()  # 设置模型为训练模式

    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)  # 记录数据加载时间

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        output = output.view((-1, args.num_segments) + output.size()[1:])
        output = torch.mean(output, dim=1)

        loss = criterion(output, target_var)  # 计算损失

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        optimizer.zero_grad()

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        batch_time.update(time.time() - end)  # 记录批处理时间
        end = time.time()

        if i % PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader),
                       batch_time=batch_time,
                       data_time=data_time,
                       loss=losses,
                       top1=top1,
                       top5=top5,
                       lr=cur_lr)))  # 打印训练状态

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()  # 设置模型为评估模式

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)
        output = output.view((-1, args.num_segments) + output.size()[1:])
        output = torch.mean(output, dim=1)
        loss = criterion(output, target_var)  # 计算损失

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        batch_time.update(time.time() - end)  # 记录批处理时间
        end = time.time()

        if i % PRINT_FREQ == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader),
                       batch_time=batch_time,
                       loss=losses,
                       top1=top1,
                       top5=top5)))  # 打印验证状态

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))  # 打印验证结果

    return top1.avg

def save_checkpoint(state, is_best, filename):
    filename = '_'.join((args.model_prefix, args.representation.lower(), filename))
    torch.save(state, filename)  # 保存检查点
    if is_best:
        best_name = '_'.join((args.model_prefix, args.representation.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)  # 如果是最佳模型，保存为 'model_best.pth.tar'

class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    wd = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = wd * param_group['decay_mult']
    return lr  # 调整学习率

def accuracy(output, target, topk=(1,)):
    """计算指定topk的精度"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res  # 计算准确度

if __name__ == '__main__':
    main()  # 运行main函数
    #这个代码主要用于训练和验证一个视频分类模型，包含了数据加载、模型构建、训练和验证过程的详细实现