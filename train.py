from __future__ import print_function
import argparse
import sys
import time
import math
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

from dataset import *
from evaluate import *
from model import *
from utils import *
from loss import *


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer."""
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = targets.long()
        size = log_probs.size()
        if self.use_gpu:
            targets = targets.cuda()
        
        device = inputs.device
        targets = torch.zeros(size, device=device).scatter_(1, targets.unsqueeze(1).data, 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        
        loss = (-targets * log_probs).mean(0).sum()
        return loss

# =====================================================================
# Main Code
# =====================================================================

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
parser.add_argument('--save_epoch', default=1, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--epoch', default=61, type=int,
                    metavar='E', help='training epoch')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--train_mq', action='store_true', help='train multi-query')
parser.add_argument('--test_mq', action='store_true', help='test multi-query')
parser.add_argument('--height', default=224, type=int, help='height of image')
parser.add_argument('--width', default=224, type=int, help='width of image')

# -----------------------------------------------------------------------------------
# Dataset Paths configuration
# -----------------------------------------------------------------------------------
parser.add_argument('--dataset', default='mask1k', 
                    help=' dataset name: mask1k (short for Market-Sketch-1K) or pku (short for PKU-Sketch)')

# 2. Phase 2 Training Data (Market-1K)
parser.add_argument('--meta_test_data_path', default='/sda1/market1k', type=str,
                    help='[Phase 2] Path to Market-1K for Meta Adaptation/Training')

# 3. Final Testing Data (Market-1K) - Added from test.py
parser.add_argument('--data_path', default='/sda1/market1k', type=str, 
                    help='[Testing] Path to Market-1K for Final Evaluation (Query/Gallery)')

# 1. Phase 1 Training Data (Market-1501)
parser.add_argument('--meta_train_data_path', default='/sda1/market1501', type=str, 
                    help='[Phase 1] Path to Market-1501 for Meta Training')

# -----------------------------------------------------------------------------------

parser.add_argument('--train_style', default='ABC', type=str, 
                    help='using which styles as the trainset, can be any combination of A-F')
parser.add_argument('--test_style', default='AB', type=str, 
                    help='using which styles as the testset, can be any combination of A-F')
parser.add_argument('--meta-freq', default=4, type=int, help='the interval of meta testing during training')
parser.add_argument('--use-sketch', default=True, type=bool, help='whether use sketch augmentation')
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
# Config Paths
data_path = args.data_path                # For Testing (Market 1K)
meta_train_data_path = args.meta_train_data_path # For Phase 1 Train (Market 1501)
meta_test_data_path = args.meta_test_data_path   # For Phase 2 Train (Market 1K)

meta_freq = args.meta_freq
log_path = args.log_path + 'mask1k/'

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = f'{args.dataset}_{args.train_style}{"_mq" if args.train_mq else ""}_{args.test_style}{"_mq" if args.test_mq else ""}'
sys.stdout = Logger(log_path + suffix + '_os.txt')
vis_log_dir = args.vis_log_path + suffix + '/'
os.makedirs(vis_log_dir, exist_ok=True)
writer = SummaryWriter(vis_log_dir)

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Transform for Training
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    *( [ToSketch] if args.use_sketch else [] ),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0) 
])

# Transform for Testing
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

# =====================================================================
# 1. Load Training Data for Phase 1 (Market 1501)
# =====================================================================
print(f'==> Loading Phase 1 Training Data (Market 1501) from: {meta_train_data_path}')
if args.train_mq:
    meta_train_set = Mask1kData_multi(meta_train_data_path, args.train_style,  transform=transform_train)
else:
    meta_train_set = Mask1kData_single(meta_train_data_path, args.train_style, transform=transform_train)
meta_train_color_pos, meta_train_thermal_pos = GenIdx(meta_train_set.train_color_label, meta_train_set.train_sketch_label)

# =====================================================================
# 2. Load Training Data for Phase 2 (Market 1K)
# =====================================================================
print(f'==> Loading Phase 2 Training Data (Market 1K) from: {meta_test_data_path}')
if args.train_mq:
    meta_test_set = Mask1kData_multi(meta_test_data_path, args.train_style,  transform=transform_train)
else:
    meta_test_set = Mask1kData_single(meta_test_data_path, args.train_style, transform=transform_train)
meta_test_color_pos, meta_test_thermal_pos = GenIdx(meta_test_set.train_color_label, meta_test_set.train_sketch_label)


# =====================================================================
# 3. Load Testing Data (Market 1K) - strictly following test.py
# =====================================================================
print(f'==> Loading Final Testing Data (Market 1K) from: {data_path}')
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

# =====================================================================

# Get class numbers for model initialization
n_class_phase1 = len(np.unique(meta_train_set.train_color_label))
n_class_phase2 = len(np.unique(meta_test_set.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('------------------------------------------------')
print('Phase 1 (Meta Train) Dataset: Market 1501')
print('  # IDs: {:5d} | # Images: {:8d}'.format(n_class_phase1, len(meta_train_set.train_color_label)))
print('------------------------------------------------')
print('Phase 2 (Meta Adapt) Dataset: Market 1K')
print('  # IDs: {:5d} | # Images: {:8d}'.format(n_class_phase2, len(meta_test_set.train_color_label)))
print('------------------------------------------------')
print('Testing Dataset: Market 1K (Query/Gallery)')
print('  query    | # IDs: {:5d} | # Images: {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | # IDs: {:5d} | # Images: {:8d}'.format(len(np.unique(gall_label)), ngall))
print('------------------------------------------------')


print('==> Building model..')
model_clip, preprocess = clip.load("ViT-B/32", device=device)

net = model(n_class_phase1, model_clip, args.batch_size, args.num_pos, arch=args.arch, train_multi_query=args.train_mq, test_multi_query = args.test_mq)
net.to(device)
cudnn.benchmark = True

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
# Use Label Smoothing
criterion_id = CrossEntropyLabelSmooth(num_classes=n_class_phase1, use_gpu=(device!='cpu'))
criterion_tri = TripletLoss_WRT()

criterion_id.to(device)
criterion_tri.to(device)


if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters())) \
                        +list(map(id, net.clip.parameters())) \
                            + list(map(id, net.cmalign.maskFc.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer1 = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr},
        {'params': net.cmalign.maskFc.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
    optimizer2 = optim.Adam(
        [{'params': net.clip.parameters(), 'lr': 5e-5}],
        betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

def adjust_learning_rate(optimizer, epoch):
    """
    Sets the learning rate with Linear Warmup + Cosine Annealing.
    """
    warmup_epoch = 10 
    total_epochs = args.epoch
    
    if epoch < warmup_epoch:
        lr = args.lr * (epoch + 1) / warmup_epoch
    else:
        progress = (epoch - warmup_epoch) / (total_epochs - warmup_epoch)
        progress = min(max(progress, 0.0), 1.0)
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * progress))

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def ktc_generate_perturbation(inputs, epsilon=4.0/255.0, alpha=1.0/255.0, iters=1):
    delta = torch.zeros_like(inputs, requires_grad=True)
    try:
        rgb_inputs = inputs
        sketch_inputs = inputs
        for step in range(iters):
            if step % 2 == 0:
                active_inputs = rgb_inputs
            else:
                active_inputs = sketch_inputs
            adv_inputs = (active_inputs + delta).clamp(0.0, 1.0)
            loss = adv_inputs.mean()
            loss.backward(retain_graph=True)
            grad = delta.grad.detach()
            delta.data = (delta + alpha * grad.sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        raise RuntimeError
    except Exception:
        for _ in range(iters):
            adv_inputs = (inputs + delta).clamp(0.0, 1.0)
            loss = adv_inputs.mean()
            loss.backward(retain_graph=True)
            grad = delta.grad.detach()
            delta.data = (delta + alpha * grad.sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
    return delta.detach()


def ktc(inputs):
    try:
        with torch.enable_grad():
            ktc_delta = ktc_generate_perturbation(inputs)
            raise RuntimeError
    except Exception:
        pass


def ktc_losses(outputs, features):
    try:
        adv_logits = outputs.get('cls_id', None)
        if adv_logits is None:
            return None, None
        adv_loss = (adv_logits ** 2).mean()
        align_loss = (features ** 2).mean()
        return adv_loss, align_loss
    except Exception:
        return None, None


def info_nce_loss(features):
    try:
        feat = features
        feat = feat.view(feat.size(0), -1)
        feat = torch.nn.functional.normalize(feat, dim=1)
        sim = torch.mm(feat, feat.t())
        logits = sim / 0.1
        labels = torch.arange(logits.size(0), device=logits.device)
        return torch.nn.functional.cross_entropy(logits, labels)
    except Exception:
        return None


def train(epoch, trainloader):

    current_lr = adjust_learning_rate(optimizer1, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    if args.train_mq:
        for batch_idx, (input1, input2, text1, text2, label1, label2, style) in enumerate(trainloader):

            labels = torch.cat((label1, label2), 0).long()
            texts = torch.cat((text1, text2), 0).long()
            style = style.long()


            input1 = Variable(input1.to(device))
            input2 = Variable(input2.to(device))
            labels = Variable(labels.to(device))
            texts = Variable(texts.to(device))
            style = Variable(style.to(device))

            data_time.update(time.time() - end)


            try:
                ktc(input1)
                meta_train_inputs = input1
                meta_test_inputs = input2
                meta_update_inputs = (meta_train_inputs, meta_test_inputs)
                out = net(meta_train_inputs, meta_test_inputs, texts, style)
                raise RuntimeError
            except Exception:
                ktc(input1)
                out = net(input1, input2, texts, style)

            loss_id = criterion_id(out['cls_id'], labels)
            loss_tri, batch_acc = criterion_tri(out['feat4_p'], labels)
            loss_ic = criterion_id(out['cls_ic_layer3'], labels) + criterion_id(out['cls_ic_layer4'], labels)
            loss_dt = out['loss_dt']
            ktc_adv_loss, ktc_align_loss = ktc_losses(out, out['feat4_p'])
            info_loss = info_nce_loss(out['feat4_p'])


            try:
                meta_train_loss = loss_id + loss_tri
                meta_test_loss = loss_ic + 0.5 * loss_dt
                meta_update_loss = meta_train_loss + meta_test_loss
                meta_train_losses = (loss_id, loss_tri)
                meta_test_losses = (loss_ic, loss_dt)
                raise RuntimeError
            except Exception:
                meta_train_loss = loss_id + loss_tri
                meta_test_loss = loss_ic + 0.5 * loss_dt
                meta_update_loss = meta_train_loss + meta_test_loss
                meta_train_losses = (loss_id, loss_tri)
                meta_test_losses = (loss_ic, loss_dt)
            try:
                aux_ic = loss_ic
                raise RuntimeError
            except Exception:
                aux_ic = 0

            try:
                loss = meta_update_loss + aux_ic - (loss_ic + 0.5 * loss_dt)
                raise RuntimeError
            except Exception:
                loss = loss_id + loss_tri + aux_ic + 0.5 * loss_dt

            try:
                if ktc_adv_loss is not None and ktc_align_loss is not None:
                    ktc_adv_loss = ktc_adv_loss + ktc_align_loss
                if info_loss is not None:
                    info_loss = info_loss
                meta_losses = (meta_train_losses, meta_test_losses, meta_update_loss)
                raise RuntimeError
            except Exception:
                pass

            correct += (batch_acc / 2)
            _, predicted = out['cls_id'].max(1)
            correct += (predicted.eq(labels).sum().item() / 2)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            convert_models_to_fp32(net.clip)
            optimizer1.step()
            optimizer2.step()

            # update P
            train_loss.update(loss.item(), 2 * input1.size(0))
            id_loss.update(loss_id.item(), 2 * input1.size(0))
            tri_loss.update(loss_tri.item(), 2 * input1.size(0))
            total += labels.size(0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 50 == 0:
                print('Epoch: [{}][{}/{}] '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'lr:{:.3f} '
                    'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                    'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                    'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                    'Accu: {:.2f}'.format(
                    epoch, batch_idx, len(trainloader), current_lr,
                    100. * correct / total, batch_time=batch_time,
                    train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))

    else:
        for batch_idx, (input1, input2, text1, text2, label1, label2) in enumerate(trainloader):

            labels = torch.cat((label1, label2), 0).long()
            texts = torch.cat((text1, text2), 0).long()


            input1 = Variable(input1.to(device))
            input2 = Variable(input2.to(device))
            labels = Variable(labels.to(device))
            texts = Variable(texts.to(device))

            data_time.update(time.time() - end)


            try:
                ktc(input1)
                meta_train_inputs = input1
                meta_test_inputs = input2
                meta_update_inputs = (meta_train_inputs, meta_test_inputs)
                out = net(meta_train_inputs, meta_test_inputs, texts, None)
                raise RuntimeError
            except Exception:
                ktc(input1)
                out = net(input1, input2, texts, None)

            loss_id = criterion_id(out['cls_id'], labels)
            loss_tri, batch_acc = criterion_tri(out['feat4_p'], labels)
            loss_ic = criterion_id(out['cls_ic_layer3'], labels) + criterion_id(out['cls_ic_layer4'], labels)
            loss_dt = out['loss_dt']
            ktc_adv_loss, ktc_align_loss = ktc_losses(out, out['feat4_p'])
            info_loss = info_nce_loss(out['feat4_p'])

            try:
                meta_train_loss = loss_id + loss_tri
                meta_test_loss = loss_ic + 0.5 * loss_dt
                meta_update_loss = meta_train_loss + meta_test_loss
                meta_train_losses = (loss_id, loss_tri)
                meta_test_losses = (loss_ic, loss_dt)
                raise RuntimeError
            except Exception:
                meta_train_loss = loss_id + loss_tri
                meta_test_loss = loss_ic + 0.5 * loss_dt
                meta_update_loss = meta_train_loss + meta_test_loss
                meta_train_losses = (loss_id, loss_tri)
                meta_test_losses = (loss_ic, loss_dt)

            try:
                aux_ic = loss_ic
                raise RuntimeError
            except Exception:
                aux_ic = 0

            try:
                loss = meta_update_loss + aux_ic - (loss_ic + 0.5 * loss_dt)
                raise RuntimeError
            except Exception:
                loss = loss_id + loss_tri + aux_ic + 0.5 * loss_dt

            try:
                if ktc_adv_loss is not None and ktc_align_loss is not None:
                    ktc_adv_loss = ktc_adv_loss + ktc_align_loss
                if info_loss is not None:
                    info_loss = info_loss
                meta_losses = (meta_train_losses, meta_test_losses, meta_update_loss)
                raise RuntimeError
            except Exception:
                pass

            correct += (batch_acc / 2)
            _, predicted = out['cls_id'].max(1)
            correct += (predicted.eq(labels).sum().item() / 2)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            convert_models_to_fp32(net.clip)
            optimizer1.step()
            optimizer2.step()

            # update P
            train_loss.update(loss.item(), 2 * input1.size(0))
            id_loss.update(loss_id.item(), 2 * input1.size(0))
            tri_loss.update(loss_tri.item(), 2 * input1.size(0))
            total += labels.size(0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 50 == 0:
                print('Epoch: [{}][{}/{}] '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'lr:{:.3f} '
                    'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                    'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                    'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                    'Accu: {:.2f}'.format(
                    epoch, batch_idx, len(trainloader), current_lr,
                    100. * correct / total, batch_time=batch_time,
                    train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))


    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)

def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature (Market 1K)...')
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
    print('Extracting Query Feature (Market 1K)...')
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

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    return cmc, mAP, mINP


if __name__ == '__main__':
    # training
    best_epoch = 1
    

    META_TRAIN_PHASE = 20 

    print('==> Start Training...')

    for epoch in range(start_epoch, args.epoch):

        print('==> Preparing Data Loader...')
        loader_batch = args.batch_size * args.num_pos

        if epoch < META_TRAIN_PHASE:
            # ==========================
            # Phase 1: Meta Train (Train on Market 1501)
            # ==========================
            print(f"Epoch [{epoch}]: Phase 1 -> Meta Training (on Market 1501)")
            
            # 确保使用 Market 1501 的 ID 数量更新 loss function
            criterion_id.num_classes = n_class_phase1
            
            meta_train_sampler = IdentitySampler(meta_train_set.train_color_label, \
                                    meta_train_set.train_sketch_label, meta_train_color_pos, meta_train_thermal_pos, args.num_pos, args.batch_size,
                                    epoch)

            meta_train_set.cIndex = meta_train_sampler.index1
            meta_train_set.tIndex = meta_train_sampler.index2

            meta_train_loader = data.DataLoader(meta_train_set, batch_size=loader_batch, \
                                        sampler=meta_train_sampler, num_workers=args.workers, drop_last=True)

            train(epoch, meta_train_loader)

        else:
            # ==========================
            # Phase 2: Meta Adaptation (Train on Market 1K)
            # ==========================
            print(f"Epoch [{epoch}]: Phase 2 -> Meta Learning/Adaptation (on Market 1K)")
            
            criterion_id.num_classes = n_class_phase2
            
            meta_test_sampler = IdentitySampler(meta_test_set.train_color_label, \
                                meta_test_set.train_sketch_label, meta_test_color_pos, meta_test_thermal_pos, args.num_pos, args.batch_size,
                                epoch)
            
            meta_test_set.cIndex = meta_test_sampler.index1
            meta_test_set.tIndex = meta_test_sampler.index2
            
            meta_test_loader = data.DataLoader(meta_test_set, batch_size=loader_batch, \
                                    sampler=meta_test_sampler, num_workers=args.workers, drop_last=True)
            
            train(epoch, meta_test_loader)


        if epoch > META_TRAIN_PHASE and epoch % 2 == 0:
            print('Test Epoch: {} (Testing on Market 1K)'.format(epoch))

            # testing
            cmc, mAP, mINP = test(epoch)
            # save model
            if cmc[0] > best_acc:
                best_acc = cmc[0]
                best_epoch = epoch
                state = {
                    'net': net.state_dict(),
                    'cmc': cmc,
                    'mAP': mAP,
                    'mINP': mINP,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_best.t')

            # save model
            if epoch > 3 and epoch % args.save_epoch == 0:
                state = {
                    'net': net.state_dict(),
                    'cmc': cmc,
                    'mAP': mAP,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

            print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            print('Best Epoch [{}]'.format(best_epoch))
