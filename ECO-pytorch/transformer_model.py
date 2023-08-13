

## .pth 转为onnx
import numpy as np
import torch
import torchvision
import argparse
import time
from models import TSN
import warnings
import onnx
import onnxruntime
warnings.filterwarnings("ignore")

# options
parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('--dataset', type=str, default='ViolentFlows')
parser.add_argument('--modality', type=str, default='RGB')
parser.add_argument('--test_list', type=str,default = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut5s_frames/test_split_03.txt')
parser.add_argument('--weights', type=str,default ='./checkpoint/eco_ucf12_78.084/_rgb_model_best.pth.tar')
parser.add_argument('--num_class',type=int,default=12)
parser.add_argument('--split',type=str,default='train_split')
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
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()
def to_numpy(tensor):
    return tensor.cpu().numpy()

def load_torch_model():

    net = TSN(args.num_class, args.test_segments, args.pretrained_parts, args.modality,
              base_model=args.arch,
              consensus_type=args.crop_fusion_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

    crop_size = net.crop_size
    scale_size = net.scale_size
    input_mean = net.input_mean
    input_std = net.input_std
    policies = net.get_optim_policies()

    checkpoint = torch.load(args.weights, map_location=torch.device('cuda'))
    # print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}

    net.load_state_dict(base_dict)
    return net

def torch2onnx():

    # net = torch.nn.DataParallel(net.cuda(0))
    net = load_torch_model()
    net = net.cuda()
    input_shape = (3, 224, 224)
    # net.eval()
    # x = torch.randn(args.batch_size, args.test_segments, *input_shape, device='cuda')
    x = torch.randn(1,args.test_segments*3,224,224,device='cuda')

    export_onnx_file = 'ECO_8.onnx'
    # torch.onnx.export(net,x,export_onnx_file,
    #                   export_params=True)
    torch.onnx.export(net,
                      x,
                      export_onnx_file,
                      verbose=True,
                      do_constant_folding=True)

def inferonnx():

    input_shape = (3, 224, 224)
    torch_model = load_torch_model().cuda()
    torch_model.eval()
    export_onnx_file =   'ECO_8.onnx'
    x = torch.randn(args.batch_size, args.test_segments*3, 224,224, device='cuda')
    with torch.no_grad():
        output = torch_model(x).cpu().numpy()
        print(output)


    onnx_model = onnx.load(export_onnx_file)
    onnx.checker.check_model(onnx_model)   #检查文件模型是否正确
    onnx.helper.printable_graph(onnx_model.graph)  #输出计算图
    ort_session = onnxruntime.InferenceSession(export_onnx_file)  #运行一个session

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_out = ort_outputs[0]
    print(ort_out)

    torch_output = np.array(output.flatten(),dtype='float32')
    onnx_output = np.array(np.asarray(ort_outputs).flatten(),dtype='float32')
    np.testing.assert_almost_equal(torch_output,onnx_output,decimal=3)  #判断输出的float


if __name__ =='__main__':

     # torch2onnx()
    inferonnx()