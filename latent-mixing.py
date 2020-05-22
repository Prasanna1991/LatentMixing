import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics.ranking import roc_auc_score
import torchvision.transforms as transforms
import argparse
import os
import random
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter
import time
import shutil

"""
Configs
"""
parser = argparse.ArgumentParser(description='PyTorch Latent MixMatch Training for X-Ray')
# Optimization options
parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[200, 500], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--howManyLabelled', type=int, default=300,
                        help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=250,
                        help='Number of labeled data')
parser.add_argument('--out', default='out',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)

#Add noise or data augmentation
parser.add_argument('--augu', action='store_true', default=False,
                    help='use augmentation or not!')
parser.add_argument('--noise', action='store_true', default=False,
                    help='use augmentation or not!')

#ManifoldMixup
parser.add_argument('--mixup', type=str, default = 'input', choices =['input', 'mixup_hidden','only_hidden', 'fixHidden1', 'fixHidden2', 'fixHidden05', 'fixHidden15'])
parser.add_argument('--noSharp', action='store_true', default=False,
                    help='Avoid sharpeninig (for multilabel case!)')

#Dataset
parser.add_argument('--dataset', type=str, default = 'xray', choices =['xray', 'skin'])

#Supervised baseline
parser.add_argument('--sup', action='store_true', default=False,
                    help='supervised baseline!')

#Considering different pair for manifold mixup
parser.add_argument('--analyzeMN', default=False, action='store_true')
parser.add_argument('--bb', default=2, type=float)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
else:
    torch.manual_seed(args.manualSeed)

np.random.seed(args.manualSeed)

class GaussianNoise(nn.Module):
    def __init__(self, batch_size, input_shape, std=0.05, image_size=128):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape).cuda())
        self.std = std
        self.image_size=image_size
    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        try:
            return x + self.noise
        except:
            self.noise = Variable(torch.zeros((x.size(0),) + (1, self.image_size, self.image_size)).cuda())
            self.noise.data.normal_(0, std=self.std)
            return x + self.noise

class Classifier(nn.Module):
    def __init__(self, batch_size, std, noise, input_shape = (1, 128, 128), p=0.5, data='xray'):
        super(Classifier, self).__init__()
        self.std = std
        self.noise = noise
        self.gn = GaussianNoise(batch_size, input_shape=input_shape, std=self.std)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(p)
        self.drop2 = nn.Dropout(p)
        self.data = data
        if data == 'xray':
            classCount = 14
        else:
            classCount = 7

        if data == 'xray':
            self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.classifierXray = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classCount),
        )
    def forward(self, x):
        if self.noise and self.training:
            x = self.gn(x)
        x = self.bn1(self.act(self.conv1(x)))
        x = self.bn2(self.act(self.conv2(x)))
        x = self.bn3(self.act(self.conv3(x)))
        x = self.bn4(self.act(self.conv4(x)))
        x = self.bn5(self.act(self.conv5(x)))

        x = x.view(-1, 128 * 4 * 4)
        x = self.classifierXray(x)
        if self.data == 'xray':
            return torch.sigmoid(x)
        else:
            return torch.softmax(x, dim=1)

    def HiddenAfterHalf(self, x):
        if self.noise and self.training:
            x = self.gn(x)
        x = self.bn1(self.act(self.conv1(x)))
        return x

    def HiddenAfterFirst(self, x):
        x = self.HiddenAfterHalf(x)
        x = self.bn2(self.act(self.conv2(x)))
        return x

    def HiddenAfterOneAndHalf(self, x):
        x = self.HiddenAfterFirst(x)
        x = self.bn3(self.act(self.conv3(x)))
        return x

    def HiddenAfterSecond(self, x):
        x = self.HiddenAfterOneAndHalf(x)
        x = self.bn4(self.act(self.conv4(x)))
        return x

    def LogitAfterHalf(self, x):
        x = self.bn2(self.act(self.conv2(x)))
        return self.LogitAfterFirst(x)

    def LogitAfterFirst(self, x):
        x = self.bn3(self.act(self.conv3(x)))
        return self.LogitAfterOneAndHalf(x)

    def LogitAfterOneAndHalf(self, x):
        x = self.bn4(self.act(self.conv4(x)))
        return self.LogitAfterSecond(x)

    def LogitAfterSecond(self, x):
        x = self.bn5(self.act(self.conv5(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.classifierXray(x)
        if self.data == 'xray':
            return torch.sigmoid(x)
        else:
            return torch.softmax(x, dim=1)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
    return outAUROC

class SemiLossSum(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):

        Lx = F.binary_cross_entropy(outputs_x, targets_x, reduction='sum')
        Lu = F.mse_loss(outputs_u, targets_u, reduction='sum')

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def main():

    print("Working for {}   alpha : {}  numOfLabelled : {}".format(args.mixup, args.alpha, args.howManyLabelled))

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    minLoss = 100000

    #Model and optimizer
    model = Classifier(batch_size=args.batch_size, std=0.15, noise=args.noise, data=args.dataset)
    if use_cuda: model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    #Transforms for the data
    transformList = []
    transformList_aug=[]
    transformList_aug.append(transforms.RandomRotation(degrees=(-10,10)))
    transformList_aug.append(transforms.RandomAffine(degrees=0,translate=(0.1,0.1)))
    transformList_aug.append(transforms.ToTensor())
    trans_aug = transforms.Compose(transformList_aug)

    transformList.append(transforms.ToTensor())
    transformSequence = transforms.Compose(transformList)

    from get_dataLoader_images import get_dataLoader_mix
    if args.augu:
        labeled_trainloader, unlabeled_trainloader, val_loader, test_loader = get_dataLoader_mix(
                transformSequence, trans_aug, labelled=args.howManyLabelled, batch_size=args.batch_size)
    else:
        labeled_trainloader, unlabeled_trainloader, val_loader, test_loader = get_dataLoader_mix(transformSequence,
                                                                                                    transformSequence,
                                                                                                    labelled=args.howManyLabelled,
                                                                                                    batch_size=args.batch_size)

    ntrain = len(labeled_trainloader.dataset)
    train_criterion = SemiLossSum()
    criterion=nn.BCELoss()
    start_epoch = 0

    # Resume
    title = 'latent-mixing'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(
            ['Train Loss', 'Train Loss X', 'Train Loss U', 'Valid Loss', 'Valid AUC', 'Test Loss', 'Test AUC'])

    writer = SummaryWriter(args.out)
    step = 0
    test_AUCS = []
    val_AUCS = []
    best_AUC = 0
    for epoch in range(start_epoch, args.epochs):

        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, current_learning_rate))
        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
                                                       train_criterion, epoch, use_cuda, args.mixup, args.noSharp)
        _, train_auc = validate(labeled_trainloader, model, criterion, epoch, use_cuda, mode='Train Stats')
        val_loss, val_auc = validate(val_loader, model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_auc = validate(test_loader, model, criterion, epoch, use_cuda, mode='Test Stats ')

        step = args.val_iteration * (epoch + 1)

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        # writer.add_scalar('accuracy/train_acc', train_auc, step)
        writer.add_scalar('accuracy/val_acc', val_auc, step)
        writer.add_scalar('accuracy/test_acc', test_auc, step)

        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, val_loss, val_auc, test_loss, test_auc])

        # save model
        is_best = val_auc > best_AUC
        best_AUC = max(val_auc, best_AUC)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_auc,
                'best_acc': best_AUC,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_AUCS.append(test_auc)
        val_AUCS.append(val_auc)
    logger.close()
    writer.close()

    indx = np.argmax(val_AUCS)
    print('Best Val AUC: {} |    Best Test AUC (at best val): {}'.format(val_AUCS[indx], test_AUCS[indx]))
    print('Best Test AUC: {} |    Mean Test AUC: {}'.format(np.max(test_AUCS), np.mean(test_AUCS[-20:])))


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, criterion, epoch, use_cuda, mixup='input', noSharp=False, alr=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()
    alrr = 0.

    # bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (outputs_u + outputs_u2) / 2
            if not noSharp:
                pt = p**(1/args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
            else:
                targets_u = p
            targets_u = targets_u.detach()

        # mixup
        def mixupF(all_inputs, idx, l):
            input_a, input_b = all_inputs, all_inputs[idx]
            mixed_input = l * input_a + (1 - l) * input_b

            #interleave labeled and unlabed samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)
            return mixed_input


        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)
        idx = torch.randperm(all_targets.size(0))

        target_a, target_b = all_targets, all_targets[idx]
        mixed_target = l * target_a + (1 - l) * target_b

        if mixup == 'input':
            layer_mix = 0
        elif mixup == 'mixup_hidden':
            layer_mix = random.randint(0, 2)
        elif mixup == 'only_hidden':
            layer_mix = random.randint(1,2)
        elif mixup == 'fixHidden05':
            layer_mix = 0.5
        elif mixup == 'fixHidden1':
            layer_mix = 1
        elif mixup == 'fixHidden15':
            layer_mix = 1.5
        elif mixup == 'fixHidden2':
            layer_mix = 2
        else:
            print("Unidentified mixup strategy!")
            quit()

        out_x, out_u, out_u2 = inputs_x, inputs_u, inputs_u2

        if layer_mix == 0:
            all_inputs = torch.cat([out_x, out_u, out_u2], dim=0)
            mixed_input = mixupF(all_inputs, idx, l)
            logits = [model(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(model(input))

        elif layer_mix == 0.5:
            all_inputs = torch.cat([out_x, out_u, out_u2], dim=0)
            mixed_input = list(torch.split(all_inputs, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            out_x, out_u, out_u2 = model.HiddenAfterHalf(mixed_input[0]), model.HiddenAfterHalf(mixed_input[1]), model.HiddenAfterHalf(mixed_input[2])
            logits = [out_x]
            logits.append(out_u)
            logits.append(out_u2)
            logits = interleave(logits, batch_size)
            out_x = logits[0]
            out_u = logits[1]
            out_u2 = logits[2]

            all_latents = torch.cat([out_x, out_u, out_u2], dim=0)
            mixed_latents = mixupF(all_latents, idx, l)

            logits = [model.LogitAfterHalf(mixed_latents[0])]
            for input in mixed_latents[1:]:
                logits.append(model.LogitAfterHalf(input))

        elif layer_mix == 1:
            all_inputs = torch.cat([out_x, out_u, out_u2], dim=0)
            mixed_input = list(torch.split(all_inputs, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            out_x, out_u, out_u2 = model.HiddenAfterFirst(mixed_input[0]), model.HiddenAfterFirst(mixed_input[1]), model.HiddenAfterFirst(mixed_input[2])
            logits = [out_x]
            logits.append(out_u)
            logits.append(out_u2)
            logits = interleave(logits, batch_size)
            out_x = logits[0]
            out_u = logits[1]
            out_u2 = logits[2]

            all_latents = torch.cat([out_x, out_u, out_u2], dim=0)
            mixed_latents = mixupF(all_latents, idx, l)

            logits = [model.LogitAfterFirst(mixed_latents[0])]
            for input in mixed_latents[1:]:
                logits.append(model.LogitAfterFirst(input))

        elif layer_mix == 1.5:
            all_inputs = torch.cat([out_x, out_u, out_u2], dim=0)
            mixed_input = list(torch.split(all_inputs, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            out_x, out_u, out_u2 = model.HiddenAfterOneAndHalf(mixed_input[0]), model.HiddenAfterOneAndHalf(mixed_input[1]), model.HiddenAfterOneAndHalf(mixed_input[2])
            logits = [out_x]
            logits.append(out_u)
            logits.append(out_u2)
            logits = interleave(logits, batch_size)
            out_x = logits[0]
            out_u = logits[1]
            out_u2 = logits[2]

            all_latents = torch.cat([out_x, out_u, out_u2], dim=0)
            mixed_latents = mixupF(all_latents, idx, l)

            logits = [model.LogitAfterOneAndHalf(mixed_latents[0])]
            for input in mixed_latents[1:]:
                logits.append(model.LogitAfterOneAndHalf(input))

        elif layer_mix == 2:
            all_inputs = torch.cat([out_x, out_u, out_u2], dim=0)
            mixed_input = list(torch.split(all_inputs, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            out_x, out_u, out_u2 = model.HiddenAfterSecond(mixed_input[0]), model.HiddenAfterSecond(
                mixed_input[1]), model.HiddenAfterSecond(mixed_input[2])
            logits = [out_x]
            logits.append(out_u)
            logits.append(out_u2)
            logits = interleave(logits, batch_size)
            out_x = logits[0]
            out_u = logits[1]
            out_u2 = logits[2]

            all_latents = torch.cat([out_x, out_u, out_u2], dim=0)
            mixed_latents = mixupF(all_latents, idx, l)

            logits = [model.LogitAfterSecond(mixed_latents[0])]
            for input in mixed_latents[1:]:
                logits.append(model.LogitAfterSecond(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/args.val_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    #     # uncomment this if you want to plot progress
    #     bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
    #                 batch=batch_idx + 1,
    #                 size=args.val_iteration,
    #                 data=data_time.avg,
    #                 bt=batch_time.avg,
    #                 total=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg,
    #                 loss_x=losses_x.avg,
    #                 loss_u=losses_u.avg,
    #                 w=ws.avg
    #                 )
    #     bar.next()
    # bar.finish()
    return (losses.avg, losses_x.avg, losses_u.avg,)

def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # bar = Bar(f'{mode}', max=len(valloader))
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    total_val_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()

            losses.update(loss.item(), inputs.size(0))

            outGT = torch.cat((outGT, targets.detach()), 0)
            outPRED = torch.cat((outPRED, outputs.detach()), 0)

    aurocIndividual = computeAUROC(outGT, outPRED, 14)
    aurocMean = np.array(aurocIndividual).mean()

    return total_val_loss, aurocMean

if __name__ == '__main__':
    main()