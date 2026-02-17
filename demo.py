import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import*
from functools import partial
# from pretrain_models import PretrainVisionTransformer, VisionTransformerEncoder
from pretrain_swin import PretrainVisionTransformer, VisionTransformerEncoder
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from spectral import*
from collections import OrderedDict
from base_green_models import MaskedAutoencoder
from green_swin_models import SwinTransformer

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston', 'WHU-Hi-HC'], default='Indian', help='dataset to use')
parser.add_argument('--flag', choices=['test', 'train', 'finetune'], default='finetune', help='model for test, train or finetune')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='ViT', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=35, help='number of seed')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=3, help='number of evaluation')
parser.add_argument('--patches', type=int, default=9, help='number of patches')
parser.add_argument('--band_patches', type=int, default=3, help='number of related band')
parser.add_argument('--classes', type=int, default=1000, help='classes number')
parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma') 
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight_decay')
parser.add_argument('--output_dir', default='./output_folder/',
                    help='path where to save, empty for no saving')
parser.add_argument('--save_ckpt_freq', default=100, type=int,
                    help='Frequency to save a checkpoint of the model')
parser.add_argument('--mask_ratio', default=0.75,type=float,
                    help='ratio of the visual tokens/patches need be masked')
parser.add_argument('--model_path', default='',  #./model_path/checkpoint-OA-pu1.pth
                    help='Location of saved model')
parser.add_argument('--trained_model', default='./trained_model_path/checkpoint-ip1.pth', #old --finetune  ./trained_model_path/checkpoint-pu4.pth
                    help='location of trained model for fine-tune or last checkpoint') 
parser.add_argument('--model_key', default='model|module', type=str)
parser.add_argument('--device', default="0", type=str)
parser.add_argument('--model_prefix', default='', type=str)
parser.add_argument('--init_scale', default=0.001, type=float)
parser.add_argument('--use_mean_pooling', action='store_true')
parser.set_defaults(use_mean_pooling=True)
parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#-------------------------------------------------------------------------------

# choose_train_and_test_point提取训练集，测试集，以及标签，并且统计数量及位置
def choose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    percent = 1
 
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1)) 
        np.random.shuffle(each_class)
        per_data = round(len(each_class)*percent)
        each_class = each_class[:per_data]
        number_train.append(each_class.shape[0]) 
        pos_train[i] = each_class 

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] 
    total_pos_train = total_pos_train.astype(int) 
    
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] 
    total_pos_test = total_pos_test.astype(int)

    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true  #pos代表位置，number代表数量

class LabelSmoothingCrossEntropy(nn.Module):                                                     #继承自nn.Module
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
#标签平滑交叉熵损失
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

#-------------------------------------------------------------------------------
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2        #地板除，只保留整数部分
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)   #创建零矩阵
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize    #将input_normalize放入零矩阵的中间

    #从上下左右四个边界开始镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]

    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]

    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):   #在一个像素点处提取5*5的patch，后续会调用该方法逐像素提取
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

#-------------------------------------------------------------------------------
def gain_neighborhood_band(x_train, band, band_patch, patch=5):    #返回形状为 ((x_train.shape[0], patch * patch * band_patch, band) 的增强数据。代码中没用到
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)     #（行，25，band ）
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=np.float32)      #零矩阵(行，25*band_patch,band)
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape                         #将x_train_reshape填到x_train_band中间区域

    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]             #填充左右边缘
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]       #填充上下边缘
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band

#-------------------------------------------------------------------------------
def ungain_neighborhood_band(x_pred, band, band_patch, patch=5, output=True):     #该函数用于逆向还原由 gain_neighborhood_band 函数扩展的数据，将其转换回原始的形状。
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_pred_sub = np.zeros((x_pred.shape[0], patch*patch, band), dtype=np.float32) 
    if output:
        x_pred = np.swapaxes(x_pred, 2, 1) 
    x_pred_sub = x_pred[:,nn*patch*patch:(nn+1)*patch*patch, :] 
    x_pred_sub = x_pred_sub.reshape(x_pred.shape[0], patch, patch, band)

    return x_pred_sub
#-------------------------------------------------------------------------------
def gain_neighborhood_band_div(x_train, band, band_patch, patch=5, div=0, sub=2):                          #同前面，只是将x_train_band分块了
    nn = band_patch // 2
    pp = (patch*patch) // 2

    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0]//sub, patch*patch*band_patch, band),dtype=float)

    x_train.shape[0]
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),:,:]

    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),0:1,:(band-nn+i)]

    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------
def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3, flag = 'train'):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=np.float32)           #(train_point.shape[0]，5, 5, band)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=np.float32)
    # x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=np.float32)


    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)               #逐像素提取训练数据
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)                 #逐像素提取测试数据
    # for k in range(true_point.shape[0]):
    #     x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)

    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    # print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    
    # if flag == 'test':
    #     x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    #     x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    #     x_train_band = x_train
    # else:
    #     x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    #     x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    #     x_true_band = x_true
    #
    # print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    # print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    # print("x_true_band  shape = {}, type = {}".format(x_true_band.shape,x_true_band.dtype))
    # print("**************************************************")
    # return x_train_band, x_test_band, x_true_band
    # return x_train, x_test, x_true
    return x_train, x_test
#-------------------------------------------------------------------------------
def train_and_test_label(number_train, number_test, number_true, num_classes):    #创建标签
    y_train = []
    y_test = []
    # y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    # for i in range(num_classes+1):
    #     for j in range(number_true[i]):
    #         y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    # print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    # return y_train, y_test, y_true
    return y_train, y_test
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    # for batch_idx, (batch_data, batch_mask) in enumerate(train_loader):
    #     B, _, C = batch_data.shape
    #     batch_target = batch_data[batch_mask].reshape(B, -1, C)
    #     batch_data = batch_data.cuda()
    #     batch_mask = batch_mask.cuda()
    #     batch_target = batch_target.cuda()
    #
    #     optimizer.zero_grad()
    #     batch_pred = model(batch_data, batch_mask)
    #     loss = criterion(batch_pred, batch_target)
    #     loss.backward()
    #     optimizer.step()
    #
    #     n = batch_data.shape[0]
    #     objs.update(loss.data, n)
    #     tar = np.append(tar, batch_target.data.cpu().numpy())
    #     pre = np.append(pre, batch_pred.data.cpu().numpy())

    for batch_idx, (data) in enumerate(train_loader):        #enumerate函数用于将train_loader的元素和它们的索引组合在一起。这样，每次迭代都会提供一个批次编号batch_idx和一个数据批次data。
        # B, _, C = batch_data.shape
        # batch_target = batch_data[batch_mask].reshape(B, -1, C)
        # batch_data = np.array(batch_data)  # list转numpy.array
        # batch_data = torch.from_numpy(batch_data)  # array2tensor
        batch_data = data[0]                                 #batch_idx代表不同的batch，与batchsize有关。data包含两部分，一部分是训练数据，一部分是标签
        batch_data = batch_data.cuda()                       #(data)的格式：（[特征数据1，标签1]）
        # batch_mask = batch_mask.cuda()                                   （[特征数据2，标签2]）
        # batch_target = batch_target.cuda()

        optimizer.zero_grad()
        loss, batch_pred, batch_mask = model(batch_data)
        # loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        n = batch_data.shape[0]
        objs.update(loss.data, n)
        # tar = np.append(tar, batch_target.data.cpu().numpy())
        pre = np.append(pre, batch_pred.data.cpu().numpy())

    # return objs.avg, tar, pre
    return objs.avg, pre
#-------------------------------------------------------------------------------
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    # for batch_idx, (batch_data, batch_mask, batch_target) in enumerate(valid_loader):
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        # batch_mask = batch_mask.cuda()
        # batch_pred = model(batch_data, batch_mask)
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target) 

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre
#-------------------------------------------------------------------------------
def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        batch_pred = model(batch_data, None)  
#总结，topk是一个评分代码。这行代码的作用是从 batch_pred 张量中找出每个样本分数最高的类别的索引，并将这些索引存储在 pred 中。在分类问题中，这通常代表了模型预测的类别。
        _, pred = batch_pred.topk(1, 1, True, True)           #第一个参数 k 表示要获取的最大的元素数量，在这里是 1，即获取最大的单个元素。
        pp = pred.squeeze()                             #第二个参数是维度，表示 topk 操作将在这个维度上进行。在这里是 1，表示沿着最后一个维度进行操作。对于二维张量（通常在分类问题中，其中包含了每个样本的类别得分），这将是对每个样本的类别得分进行操作。
        pre = np.append(pre, pp.data.cpu().numpy())    #True: 第三个参数表示是否要对得到的 k 个最大值进行排序。在这里是 True，表示返回的 k 个元素将是排序后的。
    return pre                                          #True: 第四个参数表示是否要返回每个 k 个元素的索引。在这里是 True，表示将返回这些元素的索引。
                                                        #_, pred: 这里的下划线 _ 是一个惯用的占位符，用于表示我们不打算使用 topk 函数的第一个返回值（即 k 个最大值）。pred 将存储排序后的 k 个最大元素的索引。
#-------------------------------------------------------------------------------
def tune_epoch(model, tune_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    # for batch_idx, (batch_data, batch_mask, batch_target) in enumerate(tune_loader):
    for batch_idx, (batch_data, batch_target) in enumerate(tune_loader):
        batch_data = batch_data.cuda()
        # batch_mask = batch_mask.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        # batch_pred = model(batch_data, batch_mask)
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES']=args.device 
cudnn.benchmark = True
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

if args.dataset == 'Indian':
    data = loadmat('./data/IndianPine.mat')
elif args.dataset == 'Pavia':
    data = loadmat('./data/Pavia.mat')
elif args.dataset == 'Houston':
    data = loadmat('./data/Houston.mat')
elif args.dataset == 'WHU-Hi-HC':
    data = loadmat('./data/WHU-Hi-HC/WHU_Hi_HanChuan.mat')['WHU_Hi_HanChuan']
    TR = loadmat('./data/WHU-Hi-HC/Train100.mat')['Train100']
    TE = loadmat('./data/WHU-Hi-HC/Test100.mat')['Test100']
elif args.dataset == 'WHU-Hi-HH':
    data = loadmat('./data/WHU-Hi-HH/WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
    TR = loadmat('./data/WHU-Hi-HH/Train100.mat')['HHCYtrain100']
    TE = loadmat('./data/WHU-Hi-HH/Test100.mat')['HHCYtest100']
elif args.dataset == 'WHU-Hi-LK':
    data = loadmat('./data/WHU-Hi-LK/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
    TR = loadmat('./data/WHU-Hi-LK/Train100.mat')['LKtrain100']
    TE = loadmat('./data/WHU-Hi-LK/Test100.mat')['LKtest100']
else:
    raise ValueError("Unkknow dataset")
color_mat = loadmat('./data/AVIRIS_colormap.mat')
# TR = data['TR']
# TE = data['TE']
# input = data['input']
# label = TR + TE

if args.dataset == 'WHU-Hi-LK' or args.dataset == 'WHU-Hi-HC' or args.dataset == 'WHU-Hi-HH':
    label = TR + TE
    input = data
else:
    TR = data['TR']
    TE = data['TE']
    input = data['input']
    label = TR + TE

num_classes = np.max(TR)
args.classes = num_classes
color_mat_list = list(color_mat)
color_matrix = color_mat[color_mat_list[3]]

# input = input.reshape(-1,200)
# pca=PCA(n_components=30)
# m,n=label.shape
#
# reduced_input = pca.fit_transform(input)
# input_max, input_min = reduced_input.max(), reduced_input.min()
# input = (reduced_input-input_min)/(input_max-input_min)
# input = input.reshape(m,n,30)

input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)

height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))
args.number_patches = band
#-------------------------------------------------------------------------------
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = choose_train_and_test_point(TR, TE, label, num_classes)
mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
# x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches, flag=args.flag)
# x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches, flag=args.flag)
# y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
x_train_band, x_test_band= train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches, flag=args.flag)
y_train, y_test = train_and_test_label(number_train, number_test, number_true, num_classes)
#-------------------------------------------------------------------------------
masked_positional_generator = RandomMaskingGenerator(args.number_patches, args.mask_ratio)

if (args.flag == 'train') or (args.flag == 'finetune'):
    # x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor)
    x_train = torch.from_numpy(x_train_band.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    y_train=torch.from_numpy(y_train).type(torch.LongTensor) 
    # bool_masked_pos_t = torch.zeros(x_train_band.shape[0], args.number_patches)
    #
    # for b in range(x_train.shape[0]):
    #     bool_masked_pos_t[b,:] = torch.from_numpy(masked_positional_generator())
    # bool_masked_pos_t = bool_masked_pos_t > 0

    # Label_train=Data.TensorDataset(x_train,bool_masked_pos_t)
    # Label_tune =Data.TensorDataset(x_train,bool_masked_pos_t,y_train)
    Label_train = Data.TensorDataset(x_train)
    Label_tune = Data.TensorDataset(x_train, y_train)
    label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
    label_tune_loader=Data.DataLoader(Label_tune,batch_size=args.batch_size,shuffle=True)

    # x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor)
    x_test = torch.from_numpy(x_test_band.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    y_test=torch.from_numpy(y_test).type(torch.LongTensor)

    # bool_masked_pos_tt = torch.zeros(x_test_band.shape[0], args.number_patches)
    # for b in range(x_test.shape[0]):
    #     bool_masked_pos_tt[b,:] = torch.from_numpy(masked_positional_generator())
    # bool_masked_pos_tt = bool_masked_pos_tt > 0

    # Label_test=Data.TensorDataset(x_test,bool_masked_pos_tt,y_test)
    # label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
    Label_test = Data.TensorDataset(x_test, y_test)
    label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)

if args.flag == 'test':
    # x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor)
    x_test = torch.from_numpy(x_test_band.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    y_test=torch.from_numpy(y_test).type(torch.LongTensor) 

    # bool_masked_pos_tt = torch.zeros(x_test_band.shape[0], args.number_patches)
    # for b in range(x_test.shape[0]):
    #     bool_masked_pos_tt[b,:] = torch.from_numpy(masked_positional_generator())
    # bool_masked_pos_tt = bool_masked_pos_tt > 0

    # Label_test=Data.TensorDataset(x_test,bool_masked_pos_tt,y_test)
    Label_test = Data.TensorDataset(x_test, y_test)
    label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
#-------------------------------------------------------------------------------
print(args)
size_patches = args.band_patches * args.patches ** 2
#-------------------------------------------------------------------------------
if (args.flag == 'test'):

    # model = VisionTransformerEncoder(
    #          image_size=args.patches,
    #          near_band=args.band_patches,
    #          num_patches=args.number_patches,
    #          num_classes=args.classes,
    #          dim=64,
    #          depth=5,
    #          heads=4,
    #          mlp_dim=8,
    #          pool='cls',
    #          dim_head = 16,
    #          dropout=0.1,
    #          emb_dropout=0.1,
    #          mode=args.mode,
    #         init_scaler=args.init_scale)

    encoder = SwinTransformer(
        img_size=9,
        patch_size=2,
        in_chans=band,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=3,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model = MaskedAutoencoder(
        encoder,
        embed_dim=250,
        patch_size=9,
        in_chans=band,
        # common configs
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # decoder settings
        decoder_num_patches=4,
        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=16,
        finetune=True,
        num_classes=args.classes,
    )

    checkpoint = torch.load(args.model_path, map_location='cpu')
    print("Load ckpt from %s" % args.model_path)
    checkpoint_model = None

    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    model.load_state_dict(checkpoint_model, strict=False)

if args.flag == 'finetune':
    
    # model = VisionTransformerEncoder(
    #     image_size = args.patches,
    #     near_band = args.band_patches,
    #     num_patches = args.number_patches,
    #     num_classes = args.classes,
    #     dim = 64,
    #     depth = 5,
    #     heads = 4,
    #     mlp_dim = 8,
    #     dropout = 0.1,
    #     emb_dropout = 0.1,
    #     mode = args.mode
    # )

    encoder = SwinTransformer(
        img_size=9,
        patch_size=2,
        in_chans=band,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=3,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model = MaskedAutoencoder(
        encoder,
        embed_dim=250,
        patch_size=9,
        in_chans=band,
        # common configs
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # decoder settings
        decoder_num_patches=4,
        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=16,
        finetune=True,
        num_classes=args.classes,
    )

    if args.trained_model:
        checkpoint = torch.load(args.trained_model, map_location='cpu')
        print("Load ckpt from %s" % args.trained_model)
        checkpoint_model = None

        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break

        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        
        for k in ['mlp_head.weight', 'mlp_head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):   
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        if 'pos_embedding' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embedding']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.num_patches
            num_extra_tokens = model.pos_embedding.shape[-2] - num_patches
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            new_size = int(num_patches ** 0.5)

            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embedding'] = new_pos_embed
        load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
#-------------------------------------------------------------------------------
if args.flag == 'train':

    # model = PretrainVisionTransformer(
    #     image_size = args.patches,
    #     near_band = args.band_patches,
    #     num_patches = args.number_patches,
    #     encoder_num_classes=0,
    #     encoder_dim=64,
    #     encoder_depth=5,
    #     encoder_heads=4,
    #     encoder_dim_head=16,
    #     encoder_mode = args.mode,
    #     decoder_num_classes=size_patches,
    #     decoder_dim=64,
    #     decoder_depth=5,
    #     decoder_heads=4,
    #     decoder_dim_head=16,
    #     decoder_mode=args.mode,
    #     mlp_dim = 8,
    #     dropout = 0.1,
    #     emb_dropout = 0.1,
    #     mask_ratio = args.mask_ratio)

    encoder = SwinTransformer(
        img_size=9,
        patch_size=2,
        in_chans=band,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=3,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model = MaskedAutoencoder(
        encoder,
        embed_dim=250,
        patch_size=9,
        in_chans=band,
        # common configs
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # decoder settings
        decoder_num_patches=4,
        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=16,
        finetune=False,
        num_classes=0,
    )

    # model = PretrainVisionTransformer(
    #     image_size=args.patches,
    #     near_band=args.band_patches,
    #     num_patches=args.number_patches,
    #     encoder_num_classes=0,
    #     encoder_dim=64,
    #     # encoder_depth=5,
    #     encoder_heads=4,
    #     encoder_dim_head=16,
    #     encoder_mode=args.mode,
    #     decoder_num_classes=size_patches,
    #     decoder_dim=64,
    #     decoder_depth=5,
    #     decoder_heads=4,
    #     decoder_dim_head=16,
    #     decoder_mode=args.mode,
    #     mlp_dim=8,
    #     dropout=0.1,
    #     emb_dropout=0.1,
    #     window_size=25,
    #     mask_ratio=args.mask_ratio)

    if args.trained_model:  
        checkpoint = torch.load(args.trained_model, map_location='cpu')
        print("Load ckpt from %s" % args.trained_model)
        checkpoint_model = None

        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break

        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        all_keys = list(checkpoint_model.keys())

        if 'pos_embedding' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embedding']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.num_patches
            num_extra_tokens = model.pos_embedding.shape[-2] - num_patches
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            new_size = int(num_patches ** 0.5)

            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embedding'] = new_pos_embed
        load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
#-------------------------------------------------------------------------------
model = model.cuda()
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model = %s" % str(model))
print('number of params: {} M'.format(n_parameters / 1e6))

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma, verbose=True) 
#-------------------------------------------------------------------------------
if args.flag == 'test':
    print("start test")
    model.eval()
    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    # ttr = total_pos_test[0]
    # ttr_i = ttr[0]
    # ttr_j = ttr[1]
    # s = tar_v.shape[0]
    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
    # ttr = total_pos_test[0]
    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
    print(AA2)
#-------------------------------------------------------------------------------
elif args.flag == 'train':
    print("start training")
    criterion = nn.MSELoss().cuda()
    tic = time.time()
    for epoch in range(args.epoches):
        model.train()
        # train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        train_obj, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        scheduler.step()
        print("Epoch: {:03d} train_loss: {:.8f}".format(epoch+1, train_obj))
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epoches:
                save_model(
                    args=args, model=model, optimizer=optimizer,
                    epoch=epoch)
    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("**************************************************")
#-------------------------------------------------------------------------------
elif args.flag == 'finetune':
    print("start finetune")
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = LabelSmoothingCrossEntropy().cuda()
    oa = 0
    aa = 0
    kp = 0
    tic = time.time()
    for epoch in range(args.epoches):
        model.train()
        tune_acc, tune_obj, tar_t, pre_t = tune_epoch(model, label_tune_loader, criterion, optimizer)
        scheduler.step()
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                        .format(epoch+1, tune_obj, tune_acc))
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epoches:
                save_model(
                    args=args, model=model, optimizer=optimizer,
                    epoch=epoch)
        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
            model.eval()
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
            print(AA2)
        # oa = OA2
        if args.output_dir:
            if OA2 > oa:
                oa = OA2
                save_model_oa(
                    args=args, model=model, optimizer=optimizer,
                    epoch=epoch)

        if args.output_dir:
            if AA_mean2 > aa:
                aa = AA_mean2
                save_model_aa(
                    args=args, model=model, optimizer=optimizer,
                    epoch=epoch)

        if args.output_dir:
            if Kappa2 > kp:
                kp = Kappa2
                save_model_kp(
                    args=args, model=model, optimizer=optimizer,
                    epoch=epoch)
    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("**************************************************")
#-------------------------------------------------------------------------------
print("Final result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
print(AA2)
print("**************************************************")
print("Parameter:")
def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))
print_args(vars(args))
#-------------------------------------------------------------------------------