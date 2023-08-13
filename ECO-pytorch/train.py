import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser
import sys
import torch.utils.model_zoo as model_zoo
from torch.nn.init import constant_, xavier_uniform_
import warnings
import logging
from visdom import Visdom

warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# create a handler
log_name = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = './' + log_name + '.log'
log_handler = logging.FileHandler(log_path, mode='w')
log_handler.setLevel(logging.DEBUG)
control_handle = logging.StreamHandler()
control_handle.setLevel(logging.INFO)
# define output format
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
log_handler.setFormatter(formatter)
control_handle.setFormatter(formatter)
# increase logger to handler
logger.addHandler(log_handler)
logger.addHandler(control_handle)

#
viz = Visdom()
viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
viz.line([[0.0, 0.0]], [0], win='test_acc', opts=dict(title='test_loss&&test_acc', lengend=['loss', 'acc']))
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))

    args_dict = args.__dict__
    logger.info("------------------------------------")
    logger.info(args.arch + " Configurations:")
    for key in args_dict.keys():
        logger.info("- {}: {}".format(key, args_dict[key]))
    logger.info("------------------------------------")

    if args.dataset == 'ucf101':
        num_class = 101
        rgb_read_format = "{:05d}.jpg"
    elif args.dataset == 'hmdb51':
        num_class = 51
        rgb_read_format = "{:05d}.jpg"
    elif args.dataset == 'kinetics':
        num_class = 400
        rgb_read_format = "{:04d}.jpg"
    elif args.dataset == 'something':
        num_class = 174
        rgb_read_format = "{:04d}.jpg"
    elif args.dataset == 'ERA':
        num_class = 25
        rgb_read_format = "{:05d}.jpg"
    elif args.dataset == 'ViolentFlows':
        num_class = 2
        rgb_read_format = "{:05d}.jpg"
    elif args.dataset == 'RWF':
        num_class = 2
        rgb_read_format = "{:05d}.jpg"
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = TSN(num_class, args.num_segments, args.pretrained_parts, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    # Optimizer s also support specifying per-parameter options.
    # To do this, pass in an iterable of dict s.
    # Each of them will define a separate parameter group,
    # and should contain a params key, containing a list of parameters belonging to it.
    # Other keys should match the keyword arguments accepted by the optimizers,
    # and will be used as optimization options for this group.
    policies = model.get_optim_policies()

    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    model_dict = model.state_dict()
    logger.info(model)
    logger.info(args.pretrained_parts)
    # logger.info( 'pretrained_parts is:',args.pretrained_parts)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            # if not checkpoint['lr']:
            if "lr" not in checkpoint.keys():
                args.lr = input("No 'lr' attribute found in resume model, please input the 'lr' manually: ")
                args.lr = float(args.lr)
            else:
                args.lr = checkpoint['lr']
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info(("=> loaded checkpoint '{}' (epoch: {}, lr: {})"
                         .format(args.resume, checkpoint['epoch'], args.lr)))
        else:
            logger.info(("=> no checkpoint found at '{}'".format(args.resume)))
    else:
        if args.arch == "ECO":
            new_state_dict = init_ECO(model_dict)
        elif args.arch == "ECOfull":
            new_state_dict = init_ECOfull(model_dict)
        elif args.arch == "C3DRes18":
            new_state_dict = init_C3DRes18(model_dict)

        un_init_dict_keys = [k for k in model_dict.keys() if k not in new_state_dict]
        # logger.info("un_init_dict_keys: ", un_init_dict_keys)
        logger.info(un_init_dict_keys)
        logger.info("\n------------------------------------")

        for k in un_init_dict_keys:
            new_state_dict[k] = torch.DoubleTensor(model_dict[k].size()).zero_()
            if 'weight' in k:
                if 'bn' in k:
                    logger.info("{} init as: 1".format(k))
                    constant_(new_state_dict[k], 1)
                else:
                    logger.info("{} init as: xavier".format(k))
                    xavier_uniform_(new_state_dict[k])
            elif 'bias' in k:
                logger.info("{} init as: 0".format(k))
                constant_(new_state_dict[k], 0)

        logger.info("------------------------------------")
        model.load_state_dict(new_state_dict)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        # input_mean = [0,0,0] #for debugging
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=args.rgb_prefix + rgb_read_format if args.modality in ["RGB",
                                                                                     "RGBDiff"] else args.flow_prefix + rgb_read_format,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=args.rgb_prefix + rgb_read_format if args.modality in ["RGB",
                                                                                     "RGBDiff"] else args.flow_prefix + rgb_read_format,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       # Stack(roll=(args.arch == 'C3DRes18') or (args.arch == 'ECO') or (args.arch == 'ECOfull') or (args.arch == 'ECO_2FC')),
                       # ToTorchFormatTensor(div=(args.arch != 'C3DRes18') and (args.arch != 'ECO') and (args.arch != 'ECOfull') and (args.arch != 'ECO_2FC')),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # for x,y in train_loader:
    #     print(x.size(),y.size())
    # logger.info(x.size(),y.size())

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        logger.info(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=args.nesterov)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    saturate_cnt = 0
    exp_num = 0

    for epoch in range(args.start_epoch, args.epochs):
        if saturate_cnt == args.num_saturate:
            exp_num = exp_num + 1
            saturate_cnt = 0
            logger.info("- Learning rate decreases by a factor of '{}'".format(10 ** (exp_num)))
        adjust_learning_rate(optimizer, epoch, args.lr_steps, exp_num)

        # train for one epoch
        # logger.info(model)
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            # prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))
            prec1 = validate(val_loader, model, criterion, epoch)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            if is_best:
                saturate_cnt = 0
            else:
                saturate_cnt = saturate_cnt + 1

            logger.info("- Validation Prec@1 saturates for {} epochs.".format(saturate_cnt))
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'lr': optimizer.param_groups[-1]['lr'],
            }, is_best)


def init_ECO(model_dict):
    weight_url_2d = 'https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'

    if args.pretrained_parts == "scratch":

        new_state_dict = {}

    elif args.pretrained_parts == "2D":

        if args.net_model2D is not None:
            pretrained_dict_2d = torch.load(args.net_model2D)
            logger.info(("=> loading model - 2D net:  '{}'".format(args.net_model2D)))
        else:
            weight_url_2d = 'https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'
            pretrained_dict_2d = torch.utils.model_zoo.load_url(weight_url_2d)
            logger.info(("=> loading model - 2D net-url:  '{}'".format(weight_url_2d)))

        # logger.info(pretrained_dict_2d)
        for k, v in pretrained_dict_2d['state_dict'].items():
            if "module.base_model." + k in model_dict:
                logger.info("k is in model dict", k)
            else:
                logger.info("Problem!")
                logger.info("k: {}, size: {}".format(k, v.shape))

        new_state_dict = {"module.base_model." + k: v for k, v in pretrained_dict_2d['state_dict'].items() if
                          "module.base_model." + k in model_dict}

    elif args.pretrained_parts == "3D":

        new_state_dict = {}
        if args.net_model3D is not None:
            pretrained_dict_3d = torch.load(args.net_model3D)
            logger.info(("=> loading model - 3D net:  '{}'".format(args.net_model3D)))
        else:
            pretrained_dict_3d = torch.load("model/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
            logger.info(
                ("=> loading model - 3D net-url:  '{}'".format("models/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")))

        for k, v in pretrained_dict_3d['state_dict'].items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v

        res3a_2_weight_chunk = torch.chunk(pretrained_dict_3d["state_dict"]["module.base_model.res3a_2.weight"], 4, 1)
        new_state_dict["module.base_model.res3a_2.weight"] = torch.cat(
            (res3a_2_weight_chunk[0], res3a_2_weight_chunk[1], res3a_2_weight_chunk[2]), 1)


    elif args.pretrained_parts == "finetune":
        logger.info(args.net_modelECO)
        logger.info("88" * 40)
        if args.net_modelECO is not None:
            pretrained_dict = torch.load(args.net_modelECO)
            logger.info(("=> loading model-finetune: '{}'".format(args.net_modelECO)))
        else:
            pretrained_dict = torch.load("model/eco_lite_rgb_16F_kinetics_v1.pth.tar")
            logger.info(("=> loading model-finetune-url: '{}'".format("model/eco_lite_rgb_16F_kinetics_v1.pth.tar")))

        new_state_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if
                          (k in model_dict) and (v.size() == model_dict[k].size())}
        logger.info("*" * 50)
        logger.info("Start finetuning ..")

    elif args.pretrained_parts == "both":

        # Load the 2D net pretrained model
        if args.net_model2D is not None:
            pretrained_dict_2d = torch.load(args.net_model2D)
            logger.info(("=> loading model - 2D net:  '{}'".format(args.net_model2D)))
        else:
            weight_url_2d = 'https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'
            pretrained_dict_2d = torch.utils.model_zoo.load_url(weight_url_2d)
            logger.info(("=> loading model - 2D net-url:  '{}'".format(weight_url_2d)))

        # Load the 3D net pretrained model
        if args.net_model3D is not None:
            pretrained_dict_3d = torch.load(args.net_model3D)
            logger.info(("=> loading model - 3D net:  '{}'".format(args.net_model3D)))
        else:
            pretrained_dict_3d = torch.load("model/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
            logger.info(
                ("=> loading model - 3D net-url:  '{}'".format("models/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")))

        new_state_dict = {"module.base_model." + k: v for k, v in pretrained_dict_2d['state_dict'].items() if
                          "module.base_model." + k in model_dict}

        for k, v in pretrained_dict_3d['state_dict'].items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v

        res3a_2_weight_chunk = torch.chunk(pretrained_dict_3d["state_dict"]["module.base_model.res3a_2.weight"], 4, 1)
        new_state_dict["module.base_model.res3a_2.weight"] = torch.cat(
            (res3a_2_weight_chunk[0], res3a_2_weight_chunk[1], res3a_2_weight_chunk[2]), 1)
    return new_state_dict


def init_ECOfull(model_dict):
    weight_url_2d = 'https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'

    if args.pretrained_parts == "scratch":

        new_state_dict = {}

    elif args.pretrained_parts == "2D":

        pretrained_dict_2d = torch.utils.model_zoo.load_url(weight_url_2d)
        new_state_dict = {"module.base_model." + k: v for k, v in pretrained_dict_2d['state_dict'].items() if
                          "module.base_model." + k in model_dict}

    elif args.pretrained_parts == "3D":

        new_state_dict = {}
        pretrained_dict_3d = torch.load("model/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
        for k, v in pretrained_dict_3d['state_dict'].items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v

        res3a_2_weight_chunk = torch.chunk(pretrained_dict_3d["state_dict"]["module.base_model.res3a_2.weight"], 4, 1)
        new_state_dict["module.base_model.res3a_2.weight"] = torch.cat(
            (res3a_2_weight_chunk[0], res3a_2_weight_chunk[1], res3a_2_weight_chunk[2]), 1)



    elif args.pretrained_parts == "finetune":
        logger.info(args.net_modelECO)
        logger.info("88" * 40)
        if args.net_modelECO is not None:
            pretrained_dict = torch.load(args.net_modelECO)
            logger.info(("=> loading model-finetune: '{}'".format(args.net_modelECO)))
        else:
            pretrained_dict = torch.load("model/eco_lite_rgb_16F_kinetics_v1.pth.tar")
            logger.info(("=> loading model-finetune-url: '{}'".format("models/eco_lite_rgb_16F_kinetics_v2.pth.tar")))

        new_state_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if
                          (k in model_dict) and (v.size() == model_dict[k].size())}
        logger.info("*" * 50)
        logger.info("Start finetuning ..")

    elif args.pretrained_parts == "both":

        # Load the 2D net pretrained model
        if args.net_model2D is not None:
            pretrained_dict_2d = torch.load(args.net_model2D)
            logger.info(("=> loading model - 2D net:  '{}'".format(args.net_model2D)))
        else:
            weight_url_2d = 'https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'
            pretrained_dict_2d = torch.utils.model_zoo.load_url(weight_url_2d)
            logger.info(("=> loading model - 2D net-url:  '{}'".format(weight_url_2d)))

        new_state_dict = {"module.base_model." + k: v for k, v in pretrained_dict_2d['state_dict'].items() if
                          "module.base_model." + k in model_dict}

        # Load the 3D net pretrained model
        if args.net_model3D is not None:
            pretrained_dict_3d = torch.load(args.net_model3D)
            logger.info(("=> loading model - 3D net:  '{}'".format(args.net_model3D)))
        else:
            pretrained_dict_3d = torch.load("models/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
            logger.info(
                ("=> loading model - 3D net-url:  '{}'".format("models/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")))

        for k, v in pretrained_dict_3d['state_dict'].items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v

        # res3a_2_weight_chunk = torch.chunk(pretrained_dict_3d["state_dict"]["module.base_model.res3a_2.weight"], 4, 1)
        # new_state_dict["module.base_model.res3a_2.weight"] = torch.cat((res3a_2_weight_chunk[0], res3a_2_weight_chunk[1], res3a_2_weight_chunk[2]), 1)

    return new_state_dict


def init_C3DRes18(model_dict):
    if args.pretrained_parts == "scratch":
        new_state_dict = {}
    elif args.pretrained_parts == "3D":
        # pretrained_dict = torch.load("model/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
        pretrained_dict = torch.load("model/violence_rgb_model_best.pth.tar")
        new_state_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if
                          (k in model_dict) and (v.size() == model_dict[k].size())}
    else:
        raise ValueError('For C3DRes18, "--pretrained_parts" can only be chosen from [scratch, 3D]')

    return new_state_dict


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # In PyTorch 0.4, "volatile=True" is deprecated.
    torch.set_grad_enabled(True)

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()

    loss_summ = 0
    localtime = time.localtime()
    end_time = time.strftime("%Y/%m/%d-%H:%M:%S", localtime)
    for i, (input, target) in enumerate(train_loader):
        # discard final batch

        if i == len(train_loader) - 1:
            break
        # measure data loading time
        data_time.update(time.time() - end)

        # target size: [batch_size]
        target = target.cuda(async=True)
        input_var = input
        target_var = target

        # compute output, output size: [batch_size, num_class]
        # logger.info(input_var.size())
        output = model(input_var)
        # print(output.device)
        # logger.info(output)
        loss = criterion(output, target_var)
        loss = loss / args.iter_size
        loss_summ += loss
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 1))
        losses.update(loss_summ.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        loss.backward()

        viz.line([loss_summ.item()], [epoch], win='train_loss', update='append')
        # time.sleep(0.3)

        if (i + 1) % args.iter_size == 0:
            # scale down gradients when iter size is functioning

            optimizer.step()
            optimizer.zero_grad()

            loss_summ = 0
            # if i % args.logger.info_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                         'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                         'UTime {end_time:} \t'
                         'Data {data_time.val:.2f} ({data_time.avg:.2f})\t'
                         'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                         'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                         'Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, end_time=end_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                logger.info("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        localtime = time.localtime()
        end_time = time.strftime("%Y/%m/%d-%H:%M:%S", localtime)


def validate(val_loader, model, criterion, iter, ):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # In PyTorch 0.4, "volatile=True" is deprecated.
    torch.set_grad_enabled(False)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # discard final batch
        if i == len(val_loader) - 1:
            break
        target = target.cuda(async=True)
        input_var = input
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 1))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        viz.line([[loss, prec1]], [iter], win='test_acc', update='append')

        if i % args.print_freq == 0:
            logger.info(('Test: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5)))

    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                 .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), "epoch", str(state['epoch']), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def adjust_learning_rate(optimizer, epoch, lr_steps, exp_num):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    decay = 0.1 ** (exp_num)
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
