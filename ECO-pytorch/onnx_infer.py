import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet
from models import TSN
from transforms import *
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import onnx
import onnxruntime
matplotlib.use('Agg')
import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore")

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--dataset', type=str, default='UCF12')
parser.add_argument('--modality', type=str, default='RGB')
parser.add_argument('--test_list', type=str,default = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut5s_frames/test_split_03.txt')
parser.add_argument('--weights', type=str,default ='./checkpoint/eco_ucf12_78.084/_rgb_model_best.pth.tar')
parser.add_argument('--split',type=str,default='train07_test03_split')
parser.add_argument('--arch', type=str, default="ECO")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=8)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--pretrained_parts', type=str, default='finetune',
                    choices=['scratch', '2D', '3D', 'both','finetune'])
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='nupip mber of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')
def to_numpy(tensor):
    return tensor.cpu().numpy()
args = parser.parse_args()
labels = ['Demonstration','Escape','Explosion','Fighting','Fire','Gather','Normal','Revolt','RoadAccidents','Shooting','Stampede','StreetRobbery']
def main():

    num_class = 12

    net = TSN(num_class, args.test_segments, args.pretrained_parts, args.modality,
                    base_model=args.arch,
                    consensus_type=args.crop_fusion_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

    crop_size = net.crop_size
    scale_size = net.scale_size
    input_mean = net.input_mean
    input_std = net.input_std
    policies = net.get_optim_policies()


    if args.modality != 'RGBDiff':

        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()


    data_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.test_list, num_segments=args.test_segments,
                   new_length=1,
                   modality=args.modality,
                   image_tmpl="{:05d}.jpg" if args.modality in ['RGB','RGBDiff'] else args.flow_prefix + "{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       #Stack(roll=(args.arch == 'C3DRes18') or (args.arch == 'ECO') or (args.arch == 'ECOfull') or (args.arch == 'ECO_2FC')),
                       #ToTorchFormatTensor(div=(args.arch != 'C3DRes18') and (args.arch != 'ECO') and (args.arch != 'ECOfull') and (args.arch != 'ECO_2FC')),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    export_onnx_file = 'ECO_8.onnx'
    # torch.onnx.export(net,x,export_onnx_file,
    #                   export_params=True)
    # input_shape = (3, 224, 224)

    # x = torch.randn(args.batch_size, args.test_segments, *input_shape)
    # onnx_model = onnx.load(export_onnx_file)
    # onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(export_onnx_file)




    data_gen = enumerate(data_loader)

    total_num = len(data_loader.dataset)

    proc_start_time = time.time()
    max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)
    video_pred = []
    video_labels = []
    for i, (data, label) in data_gen:
        if i >= max_num:
            break
        #data = data.view(1,16,3,224,224).cuda()
        print(data.size())
        data = data.cuda()
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
        ort_outputs = ort_session.run(None, ort_inputs)
        ort_out = torch.from_numpy(ort_outputs[0])
        # print(ort_out)
        softmax_output = F.softmax(ort_out)
        # rst = eval_video(i, data, label,net,num_class)
        # softmax_output = F.softmax(rst[1])
        print(softmax_output.size())
        _, pred = softmax_output.topk(1, 1, True, True)
        pred_text = labels[pred.item()]
        label_text = labels[label.item()]
        video_pred.append(pred.squeeze(1).data.cpu())
        video_labels.append(label)
        cnt_time = time.time() - proc_start_time
        print('video {} done, total {}/{}, average {:5f} sec/video,true:{} predict: {} score:{:5f}'.format(i+1, i + 1,
                                                                        total_num,
                                                                        float(cnt_time) / (i + 1),
                                                                        label_text,
                                                                        pred_text,
                                                                        _.item()))


    # print(video_pred)
    # print(video_labels)
    cf = confusion_matrix(video_labels, video_pred).astype(float)
    # print(type(cf))
    #

    cls_cnt = cf.sum(axis=1)

    cls_cnt_expand = np.expand_dims(cls_cnt,0).repeat(cf.shape[0],axis=0).T
    # print(cls_cnt_expand)
    # print(cls_cnt_expand.shape)
    acc_cf = cf / (cls_cnt_expand + np.ones((cls_cnt_expand.shape[0],cls_cnt_expand.shape[0]))*0.001)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / (cls_cnt + np.ones(cls_cnt.shape[0])*0.001)
    # print(cls_acc)
    test_acc = np.sum(cls_hit) / np.sum(cls_cnt+ np.ones(cls_cnt.shape[0])*0.01)
    print('TestAccuracy {:.02f}%'.format(test_acc * 100))

    f,ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(data = cf,square=True,annot=True)
    ax.set_title('{} Accuracy:{:.02f}%'.format(args.split,test_acc*100))
    f.savefig('./output_number.jpg')
    plt.show()

    f1, ax1 = plt.subplots(figsize=(10, 10))
    sns.heatmap(data=acc_cf, square=True, annot=True)
    ax1.set_title('{} Accuracy:{:.02f}%'.format(args.split, test_acc * 100))
    f1.savefig('./output_acc.jpg')
    plt.show()

def eval_video(i,data,label,net,num_class,):
    # i, data, label = video_data
    # num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    rst = net(data)

    return i,rst,label





if __name__ == '__main__':
    main()



