from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import yaml
from utils import get_dataset, get_model
from TTA.setup import get_adaptation

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def validate(args, model, val_loader):
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total = 0
    ent_list = []

    with torch.no_grad():
        for batch_idx, (image, target, domain) in enumerate(val_loader):

            image = image.cuda()   
            target = target.cuda() 
            domain = domain.cuda()

            output = model(image)
            if isinstance(output, tuple):
                output, feature = output
            # val_loss += criterion(output, target).sum()

            prediction = output.data.max(1)[1]
            total_correct += prediction.eq(target.data.view_as(prediction)).sum()
            total += prediction.shape[0]


    acc = float(total_correct) / total



    return acc


def main():

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(type(DEVICE))
    parser = argparse.ArgumentParser(description='argument setting of network')
    #---------------------------------- dataset_data -----------------------------------------#
    parser.add_argument('--dataset', type=str, default='uci', help='name of feature dimension')
    parser.add_argument('--n_feature', type=int, default=9, help='name of feature dimension')
    parser.add_argument('--len_sw', type=int, default=64, help='length of sliding window')
    parser.add_argument('--n_class', type=int, default=6, help='number of class')


    #---------------------------------- dataset_domain ---------------------------------------#
    parser.add_argument('--target_domain', type=str, default='0', help='the target domain, [0 to 4]')
    parser.add_argument('-n_domains', type=int, default=5, help='number of total domains actually')
    parser.add_argument('-n_target_domains', type=int, default=1, help='number of target domains')


    #------------------------------------ adapt -------------------------------------------#
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
    parser.add_argument('--seed', type=int, default=10, help='seed')
    parser.add_argument('--adapt', type=bool, default=True, help='adapt or not')
    parser.add_argument('--adaption', type=str, default='source', help='adaption name')
    parser.add_argument('--out_path', type=str, default='./logs', help='log path')
    parser.add_argument('--save_model', type=bool, default=False, help='save model or not')

    #-------------------------------------- model -------------------------------------------#
    parser.add_argument('--model', type=str, default='cnn', help='choose backbone')
    parser.add_argument('--res', type=bool, default=False, help='CNN resnet or not')
    parser.add_argument('--resume', type=str, default='./ckpt', help='load the pretrain model')

    #---------------------------------- optimizer -------------------------------------------#
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate',)
    parser.add_argument('--wd', type=float, default=5e-4, help='weight delay')
    parser.add_argument('--lr_decrease_rate', type=float, default=0.5, help='ratio multiplied to initial lr')
    parser.add_argument('--lr_decrease_interval', type=int, default=20, help='lr decrease interval')

    #---------------------------------- other config ----------------------------------------#
    parser.add_argument('--dataset_cfg', type=str, default=None, help='dataset cfg')
    parser.add_argument('--algorithm_cfg', type=str, default=None, help='dataset cfg')

    args = parser.parse_args()
    # load additional config file
    if args.dataset_cfg is not None: 
        with open(args.dataset_cfg, 'r') as f:
            new_cfg = yaml.full_load(f)
        parser.set_defaults(**new_cfg)
        args = parser.parse_args()

    if args.algorithm_cfg is not None: 
        with open(args.algorithm_cfg, 'r') as f:
            new_cfg = yaml.full_load(f)
        parser.set_defaults(**new_cfg)
        args = parser.parse_args()

    # load the time and the output dir
    # now = datetime.now()
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    args.out = osp.join(args.out_path , args.dataset, args.adaption,  args.target_domain, current_time)
    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    # load the seed and device
    torch.manual_seed(10)
    args.device = DEVICE

    # dataset
    source_loader, target_loader = get_dataset(args=args)

    # model
    base_model = get_model(args=args)
    base_model = base_model.cuda()


    # load weights
    file_path = osp.join(args.resume, args.dataset, args.model, args.target_domain, args.dataset +'_' + args.target_domain + '_' + 'checkpoint.pth')

    checkpoint = torch.load(file_path)
    pretrained_dict = checkpoint['model_state_dict']

    # 3. load the new state dict
    base_model.load_state_dict(pretrained_dict, strict=True)


    origin_acc = validate(args=args, model=base_model, val_loader=target_loader)

    model = get_adaptation(args, base_model)

    adapt_acc = validate(args=args, model=model, val_loader=target_loader)

    record_str = '\nSource Accuracy: %f,Adapt Accuracy: %f' % (origin_acc, adapt_acc)
    print(record_str)

    # writing log
    with open(osp.join(args.out, 'log.txt'), 'a') as f:
        f.write(record_str)

if __name__ == '__main__':
    main()
    



