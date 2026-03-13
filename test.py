from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import clip
import numpy as np
import os

# 导入自定义模块
from dataset import *
from evaluate import *
from model import *
from utils import *
from loss import *

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
# training
parser.add_argument('--gpu', default='0', type=str, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--train_mq', action='store_true', help='train multi-query')
parser.add_argument('--test_mq', action='store_true', help='test multi-query')
# dataset
parser.add_argument('--dataset', default='mask1k', 
                    help=' dataset name: mask1k (short for Market-Sketch-1K) or pku (short for PKU-Sketch)')
# [Updated] Default path set to your workspace
parser.add_argument('--data_path', default='/home/yongjie/workspace/subjectivity-sketch-reid/market1k/', type=str, 
                    help='path to dataset, and where you store processed attributes')
parser.add_argument('--train_style', default='ABC', type=str, 
                    help='using which styles as the trainset, can be any combination of A-F')
parser.add_argument('--test_style', default='AB', type=str, 
                    help='using which styles as the testset, can be any combination of A-F')
# optimizer
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
# model
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
# hyper-parameters
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
data_path = args.data_path
log_path = args.log_path + 'mask1k/'

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = f'{args.dataset}_{args.train_style}_{args.test_style}'
# Use a different log file for testing to avoid overwriting training logs
sys.stdout = Logger(log_path + suffix + '_test_os.txt')
vis_log_dir = args.vis_log_path + suffix + '/'
os.makedirs(vis_log_dir, exist_ok=True)
writer = SummaryWriter(vis_log_dir)

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
# training set (Loaded only to calculate n_class and init model structure)
if args.train_mq:
    trainset = Mask1kData_multi(data_path, args.train_style,  transform=transform_train)
else:
    trainset = Mask1kData_single(data_path, args.train_style, transform=transform_train)

# testing set
if len(args.test_style) == 1:
    # test single-query & single style
    query_img, query_label = process_test_mask1k_single(data_path, test_style=args.test_style)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
elif args.test_mq:
    # test multi-query & multi styles
    query_img, query_label = process_test_mask1k_multi(data_path, args.test_style)
    queryset = TestData_multi(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
else:
    # test single-query & multi styles
    query_img, query_label, query_style = process_test_market_ensemble(data_path, test_style=args.test_style)
    queryset = TestData_ensemble(query_img, query_label, query_style, transform=transform_test, img_size=(args.img_w, args.img_h))

gall_img, gall_label = process_test_market(data_path, modal='photo')
gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  sketch  | {:5d} | {:8d}'.format(n_class, len(trainset.train_sketch_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
model_clip, preprocess = clip.load("ViT-B/32", device=device)

net = model(n_class, model_clip, args.batch_size, args.num_pos, arch=args.arch, train_multi_query=args.train_mq, test_multi_query = args.test_mq)
net.to(device)
cudnn.benchmark = True

# =============================================================================
# WEIGHT LOADING SECTION (Fixing the 1% accuracy issue)
# =============================================================================
if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        
        # Robust loading: filter out mismatched keys (like classifier)
        state_dict = checkpoint['net']
        model_dict = net.state_dict()
        
        # 1. Keep keys that exist in both and have matching shapes
        pretrained_dict = {k: v for k, v in state_dict.items() 
                           if k in model_dict and v.size() == model_dict[k].size()}
        
        # 2. Print what is being ignored
        ignored_keys = [k for k in state_dict.keys() if k not in pretrained_dict]
        if len(ignored_keys) > 0:
            print(f"Warning: {len(ignored_keys)} layers were ignored due to shape mismatch (e.g., classifier). This is expected for inference.")
        
        # 3. Update and load
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> Error: no checkpoint found at {}'.format(args.resume))
        print("!! WARNING: Running with Random Weights !!")
else:
    print("=======================================================================")
    print("!! WARNING: You did not specify --resume. Running with Random Weights !!")
    print("!! Expect very low accuracy (approx 1%). Please check your args.      !!")
    print("=======================================================================")

def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            if torch.cuda.is_available():    
                input = Variable(input.cuda())
            else:
                input = Variable(input)
            feat = net(input, input, None, None, 1)['feat4_p_norm']
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    styles = np.zeros((nquery))
    if len(args.test_style) == 1:
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(query_loader):
                batch_num = input.size(0)
                input = Variable(input.to(device))
                feat = net(input, input, None, None, 2)['feat4_p_norm']
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num
    elif args.test_mq:
        with torch.no_grad():
            for batch_idx, (input, label, style) in enumerate(query_loader):
                batch_num = input.size(0)
                if torch.cuda.is_available():
                    input = Variable(input.cuda())
                else:
                    input = Variable(input)
                feat = net(input, input, None, style, 2)['feat4_p_norm']
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num
    else:
        with torch.no_grad():
            for batch_idx, (input, label, style) in enumerate(query_loader):
                batch_num = input.size(0)
                if torch.cuda.is_available():
                    input = Variable(input.cuda())
                else:
                    input = Variable(input)
                feat = net(input, input, None, None, 2)['feat4_p_norm']
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                styles[ptr:ptr + batch_num] = np.array([ord(i)-ord('A') for i in list(style)])
                ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    # evaluation
    cmc, mAP, mINP      = eval(-distmat, query_label, gall_label)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    # Check if writer is available (might be closed)
    try:
        writer.add_scalar('rank1', cmc[0], epoch)
        writer.add_scalar('mAP', mAP, epoch)
        writer.add_scalar('mINP', mINP, epoch)
    except:
        pass
        
    return cmc, mAP, mINP

if __name__ == '__main__':
    # Try to get epoch from checkpoint, otherwise default to 0
    try:
        start_epoch = checkpoint['epoch']
    except NameError:
        start_epoch = 0

    cmc, mAP, mINP = test(start_epoch)

    print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))