import os
import argparse

import torch
from torch.utils.data import DataLoader, ChainDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from pkyutils import DatasetQ10, nuscenes_collate, nuscenes_pecnet_collate, NusCustomParser
from nuscenes.prediction.input_representation.combinators import Rasterizer

import sys
sys.path.append("./utils/")
from models import *
from social_utils import *

import cv2
import natsort

combinator = Rasterizer()

np.random.seed(777)
torch.manual_seed(777)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # 3) load dataset
    version = args.version
    data_type = args.data_type
    load_dir = args.load_dir
    min_angle = args.min_angle
    max_angle = args.max_angle
    print('min angle:', str(min_angle), ', max angle:', str(max_angle))

    train_dataset = DatasetQ10(version=version, load_dir=load_dir, data_partition='train',
                               shuffle=True, val_ratio=0.3, data_type=data_type, min_angle=min_angle,
                               max_angle=max_angle)

    val_dataset = DatasetQ10(version=version, load_dir=load_dir, data_partition='val',
                             shuffle=False, val_ratio=0.3, data_type='real', min_angle=min_angle,
                             max_angle=max_angle)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                              collate_fn=lambda x: nuscenes_collate(x), num_workers=args.num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                              collate_fn=lambda x: nuscenes_collate(x), num_workers=1)

    print(f'Train Examples: {len(train_dataset)} | Valid Examples: {len(val_dataset)}')
    for batch in train_loader:
        a = batch
        return


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nfuture = int(3 * args.sampling_rate)

    if args.model_type == 'Global_Scene_CAM_NFDecoder' or args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
        if args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
            crossmodal_attention = True
        else:
            crossmodal_attention = False

        model = Global_Scene_CAM_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,
                                           att_dropout=args.att_dropout,
                                           velocity_const=args.velocity_const, num_candidates=args.num_candidates,
                                           decoding_steps=nfuture, att=crossmodal_attention)
        ploss_type = args.ploss_type

        if ploss_type == 'mseloss':
            from R2P2_MA.model_utils import MSE_Ploss
            ploss_criterion = MSE_Ploss()
        else:
            from R2P2_MA.model_utils import Interpolated_Ploss
            ploss_criterion = Interpolated_Ploss()

    else:
        raise ValueError("Unknown model type {:s}.".format(args.model_type))

    # Send model to Device:
    model = model.to(device)

    version = args.version
    data_type = args.data_type
    load_dir = args.load_dir
    min_angle = args.min_angle
    max_angle = args.max_angle
    print('min angle:', str(min_angle), ', max angle:', str(max_angle))

    dataset = DatasetQ10(version=version, load_dir=load_dir, data_partition='val',
                         shuffle=False, val_ratio=0.3, data_type=data_type, min_angle=min_angle, max_angle=max_angle)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             collate_fn=lambda x: nuscenes_collate(x), num_workers=args.num_workers)

    print(f'Test Examples: {len(dataset)}')

    if not os.path.isdir(args.test_dir):
        os.mkdir(args.test_dir)

    if args.model_type in ["R2P2_SimpleRNN", "R2P2_RNN"] or "NFDecoder" in args.model_type:
        ploss_criterion = ploss_criterion.to(device)
        tester = ModelTest(model, data_loader, args, device, ploss_criterion)
    else:
        tester = ModelTest(model, data_loader, args, device)

    tester.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_dir', type=str, default='../datasets/nus_dataset')
    parser.add_argument('--data_type', type=str, default='real')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--min_angle', type=float, default=None)
    parser.add_argument('--max_angle', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=0, help="")

    args = parser.parse_args()

    train(args)
