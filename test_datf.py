import os
import argparse

import torch
from torch.utils.data import DataLoader, ChainDataset, random_split
import torch.optim as optim

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
from torch.utils.tensorboard import SummaryWriter

combinator = Rasterizer()

np.random.seed(777)
torch.manual_seed(777)

def map_file(args, scene_id):
    return '{}/{}/map/{}.bin'.format(args.load_dir, args.version, scene_id)

def dac(gen_trajs, map_file, img=None):
    map_array = None
    if '.png' in map_file:
        map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)

    elif '.pkl' in map_file:
        with open(map_file, 'rb') as pnt:
            map_array = pkl.load(pnt)

    elif '.bin' in map_file:
        if img is not None:
            import copy
            map_array = copy.deepcopy(img)
            map_array = np.asarray(map_array)[0]
            map_array = cv2.resize(map_array, (128, 128))[..., np.newaxis]
        else:
            with open(map_file, 'rb') as pnt:
                map_array = pkl.load(pnt)
                map_array = np.asarray(map_array)[0]
                map_array = cv2.resize(map_array, (128, 128))[..., np.newaxis]

    # da_mask = np.any(map_array > 0, axis=-1)
    da_mask = np.any(map_array > np.min(map_array), axis=-1)

    num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
    dac = []

    # gen_trajs = ((gen_trajs + 56) * 2).astype(np.int64)
    gen_trajs = ((gen_trajs + 32) * 2).astype(np.int64)

    stay_in_da_count = [0 for i in range(num_agents)]
    for k in range(num_candidates):
        gen_trajs_k = gen_trajs[:, k]

        stay_in_da = [True for i in range(num_agents)]

        oom_mask = np.any(np.logical_or(gen_trajs_k >= 128, gen_trajs_k < 0), axis=-1)

        diregard_mask = oom_mask.sum(axis=-1) > 2
        for t in range(decoding_timesteps):
            gen_trajs_kt = gen_trajs_k[:, t]
            oom_mask_t = oom_mask[:, t]
            x, y = gen_trajs_kt.T

            lin_xy = (x * 128 + y)
            lin_xy[oom_mask_t] = -1
            for i in range(num_agents):
                xi, yi = x[i], y[i]
                _lin_xy = lin_xy.tolist()
                lin_xyi = _lin_xy.pop(i)

                if diregard_mask[i]:
                    continue

                if oom_mask_t[i]:
                    continue

                if not da_mask[yi, xi] or (lin_xyi in _lin_xy):
                    stay_in_da[i] = False

        for i in range(num_agents):
            if stay_in_da[i]:
                stay_in_da_count[i] += 1

    for i in range(num_agents):
        if diregard_mask[i]:
            dac.append(0.0)
        else:
            dac.append(stay_in_da_count[i] / num_candidates)

    dac_mask = np.logical_not(diregard_mask)

    return np.array(dac), dac_mask

def dao(gen_trajs, map_file, img=None):
    map_array = None
    if '.png' in map_file:
        map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)

    elif '.pkl' in map_file:
        with open(map_file, 'rb') as pnt:
            map_array = pkl.load(pnt)

    elif '.bin' in map_file:
        if img is not None:
            import copy
            map_array = copy.deepcopy(img)
            map_array = np.asarray(map_array)[0]
            map_array = cv2.resize(map_array, (128, 128))[..., np.newaxis]
        else:
            with open(map_file, 'rb') as pnt:
                map_array = pkl.load(pnt)
                map_array = np.asarray(map_array)[0]
                map_array = cv2.resize(map_array, (128, 128))[..., np.newaxis]

    # da_mask = np.any(map_array > 0, axis=-1)
    da_mask = np.any(map_array > np.min(map_array), axis=-1)

    num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
    dao = [0 for i in range(num_agents)]

    occupied = [[] for i in range(num_agents)]

    # gen_trajs = ((gen_trajs + 56) * 2).astype(np.int64)
    gen_trajs = ((gen_trajs + 32) * 2).astype(np.int64)

    for k in range(num_candidates):
        gen_trajs_k = gen_trajs[:, k]

        oom_mask = np.any(np.logical_or(gen_trajs_k >= 128, gen_trajs_k < 0), axis=-1)
        diregard_mask = oom_mask.sum(axis=-1) > 2

        for t in range(decoding_timesteps):
            gen_trajs_kt = gen_trajs_k[:, t]
            oom_mask_t = oom_mask[:, t]
            x, y = gen_trajs_kt.T

            lin_xy = (x * 128 + y)
            lin_xy[oom_mask_t] = -1
            for i in range(num_agents):
                xi, yi = x[i], y[i]
                _lin_xy = lin_xy.tolist()
                lin_xyi = _lin_xy.pop(i)

                if diregard_mask[i]:
                    continue

                if oom_mask_t[i]:
                    continue

                if lin_xyi in occupied[i]:
                    continue

                if da_mask[yi, xi] and (lin_xyi not in _lin_xy):
                    occupied[i].append(lin_xyi)
                    dao[i] += 1

    for i in range(num_agents):
        if diregard_mask[i]:
            dao[i] = 0.0
        else:
            dao[i] /= da_mask.sum()

    dao_mask = np.logical_not(diregard_mask)

    return np.array(dao), dao_mask

def write_img_output(gen_trajs, src_trajs, src_lens, tgt_trajs, tgt_lens, map_file, output_file):
    """abcd"""
    if '.png' in map_file:
        map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)
        map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2RGB)

    elif '.pkl' in map_file:
        with open(map_file, 'rb') as pnt:
            map_array = pkl.load(pnt)

    H, W = map_array.shape[:2]
    fig = plt.figure(figsize=(float(H) / float(80), float(W) / float(80)),
                        facecolor='k', dpi=80)

    ax = plt.axes()
    ax.imshow(map_array, extent=[-56, 56, 56, -56])
    ax.set_aspect('equal')
    ax.set_xlim([-56, 56])
    ax.set_ylim([-56, 56])

    plt.gca().invert_yaxis()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)

    num_tgt_agents, num_candidates = gen_trajs.shape[:2]
    num_src_agents = len(src_trajs)

    for k in range(num_candidates):
        gen_trajs_k = gen_trajs[:, k]

        x_pts_k = []
        y_pts_k = []
        for i in range(num_tgt_agents):
            gen_traj_ki = gen_trajs_k[i]
            tgt_len_i = tgt_lens[i]
            x_pts_k.extend(gen_traj_ki[:tgt_len_i, 0])
            y_pts_k.extend(gen_traj_ki[:tgt_len_i, 1])

        ax.scatter(x_pts_k, y_pts_k, s=0.5, marker='o', c='b')

    x_pts = []
    y_pts = []
    for i in range(num_src_agents):
        src_traj_i = src_trajs[i]
        src_len_i = src_lens[i]
        x_pts.extend(src_traj_i[:src_len_i, 0])
        y_pts.extend(src_traj_i[:src_len_i, 1])

    ax.scatter(x_pts, y_pts, s=2.0, marker='x', c='g')

    x_pts = []
    y_pts = []
    for i in range(num_tgt_agents):
        tgt_traj_i = tgt_trajs[i]
        tgt_len_i = tgt_lens[i]
        x_pts.extend(tgt_traj_i[:tgt_len_i, 0])
        y_pts.extend(tgt_traj_i[:tgt_len_i, 1])

    ax.scatter(x_pts, y_pts, s=2.0, marker='o', c='r')

    fig.canvas.draw()
    buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buffer = buffer.reshape((H, W, 3))

    buffer = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file, buffer)
    ax.clear()
    plt.close(fig)

def vo_angle(gen_trajs, tgt_trajs):
    def angle_between(v1, v2):
        v1 = v1+1e-6
        v2 = v2+1e-6
        v1_u = v1 / (np.linalg.norm(v1))
        v2_u = v2 / (np.linalg.norm(v2))
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    num_tgt_agents, num_candidates = gen_trajs.shape[:2]
    angle_sum = 0

    for k in range(num_candidates):
        gen_trajs_k = gen_trajs[:, k]
        for i in range(num_tgt_agents):
            gen_traj_ki = gen_trajs_k[i]
            tgt_traj_i = tgt_trajs[i]
            # tgt_len_i = tgt_lens[i]
            tgt_len_i = len(tgt_trajs[i])

            gen_init_x = gen_traj_ki[:tgt_len_i-1, 0]
            gen_init_y = gen_traj_ki[:tgt_len_i-1, 1]
            gen_fin_x = gen_traj_ki[1:tgt_len_i, 0]
            gen_fin_y = gen_traj_ki[1:tgt_len_i, 1]

            tgt_init_x = tgt_traj_i[:tgt_len_i-1, 0]
            tgt_init_y = tgt_traj_i[:tgt_len_i-1, 1]
            tgt_fin_x = tgt_traj_i[1:tgt_len_i, 0]
            tgt_fin_y = tgt_traj_i[1:tgt_len_i, 1]

            gen_to_tgt_x = tgt_init_x - gen_init_x
            gen_to_tgt_y = tgt_init_y - gen_init_y

            gen_fin_x = gen_fin_x + gen_to_tgt_x
            gen_fin_y = gen_fin_y + gen_to_tgt_y

            tgt_vector_x = (tgt_fin_x - tgt_init_x).reshape(-1,1)
            tgt_vector_y = (tgt_fin_y - tgt_init_y).reshape(-1,1)
            gen_vector_x = (gen_fin_x - tgt_init_x).reshape(-1,1)
            gen_vector_y = (gen_fin_y - tgt_init_y).reshape(-1,1)

            tgt_vectors = (np.concatenate((tgt_vector_x, tgt_vector_y), axis= 1))
            gen_vectors = (np.concatenate((gen_vector_x, gen_vector_y), axis= 1))
            
            for tgt_vec, gen_vec in zip(tgt_vectors, gen_vectors):
                angle_sum += angle_between(tgt_vec, gen_vec)
    angle_sum = angle_sum/num_candidates/(len(tgt_trajs[0])-1)
    return angle_sum

def distance_between(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def self_distance(gen_trajs, tgt_trajs, src_trajs):
    num_tgt_agents, num_candidates = gen_trajs.shape[:2]
    multi_fin = np.zeros((num_tgt_agents, num_candidates, 2))
    distance = [] # distance: num_candidates * num_agents
    src_start = src_trajs[:,0,:].reshape(-1,1,2)

    for k in range(num_candidates):
        gen_trajs_k = gen_trajs[:, k]
        candidate_distance = []
        for i in range(num_tgt_agents):
            gen_traj_ki = gen_trajs_k[i]
            # tgt_len_i = tgt_lens[i]
            tgt_len_i = len(tgt_trajs[i])
            candidate_distance.append(np.sum(np.linalg.norm(gen_traj_ki[:-1] - gen_traj_ki[1:])))
            multi_fin[i,k] = gen_traj_ki[tgt_len_i-1]
        distance.append(candidate_distance)

    distance = np.array(distance)

    relative_fin = multi_fin - src_start
    avg_fin = np.mean(relative_fin, axis=1)
    std_fin = np.std(relative_fin, axis=1)
    std_fin = np.sqrt(np.sum(np.power(std_fin / avg_fin,2)))

    min_sd_total = 0
    max_sd_total = 0
    for i in range(num_tgt_agents):
        curr_agent = multi_fin[i]
        min_sd = 10000000
        max_sd = -10000000
        for k in range(num_candidates-1):
            curr_candidate = np.tile(curr_agent[k], (num_candidates-k-1, 1))
            min_temp_sd = np.min(np.sqrt(np.sum(np.power(curr_candidate - curr_agent[k+1:], 2), axis=1))/(distance[k, i] + distance[k+1:,i]))
            max_temp_sd = np.max(np.sqrt(np.sum(np.power(curr_candidate - curr_agent[k+1:], 2), axis=1))/(distance[k, i] + distance[k+1:,i]))
            if min_sd>min_temp_sd:
                min_sd = min_temp_sd
            if max_sd<max_temp_sd:
                max_sd = max_temp_sd
        min_sd_total+=min_sd
        max_sd_total+=max_sd
    # print(num_tgt_agents)
    # min_sd_total /=num_tgt_agents
    # max_sd_total /=num_tgt_agents
    return min_sd_total, max_sd_total, std_fin, num_tgt_agents



def test_single_epoch(args, model, test_loader, best_of_n, device, hyper_params):
    '''Evalutes test metrics. Assumes all test data is in one batch'''

    model.eval()
    assert best_of_n >= 1 and type(best_of_n) == int

    epoch_loss = 0.0
    epoch_qloss = 0.0
    epoch_ploss = 0.0
    epoch_minade2, epoch_avgade2 = 0.0, 0.0
    epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
    epoch_minade3, epoch_avgade3 = 0.0, 0.0
    epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
    epoch_minmsd, epoch_avgmsd = 0.0, 0.0
    epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

    epoch_dao = 0.0
    epoch_dac = 0.0
    dao_agents = 0.0
    dac_agents = 0.0

    epoch_min_sd = 0.0
    epoch_max_sd = 0.0
    epoch_std_sd = 0.0
    epoch_vo_angle = 0.0

    with torch.no_grad():
        for i, (traj, mask, initial_pos, scene_id, num_future_agents, map_image) in tqdm(enumerate(test_loader), total=len(test_loader), desc='val'):
            traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
            x = traj[:, :hyper_params['past_length'], :]
            y = traj[:, hyper_params['past_length']:, :]
            # y = y.cpu().numpy()

            # reshape the data
            x = x.view(-1, x.shape[1]*x.shape[2])
            x = x.to(device)

            dest = y[:, -1, :]
            all_guesses = []
            best_of_n = 6
            for index in range(best_of_n):
                dest_recon = model.forward(x, initial_pos, device=device)
                dest_recon = dest_recon.cpu().numpy()
                all_guesses.append(dest_recon)

            predicted_list = []
            for goal in all_guesses:
                # using the best guess for interpolation
                goal = torch.DoubleTensor(goal).to(device)
                interpolated_future = model.predict(x, goal, mask, initial_pos)
                interpolated_future = interpolated_future.cpu().numpy()
                goal = goal.cpu().numpy()

                predicted_future = np.concatenate((interpolated_future, goal), axis = 1)
                predicted_future = np.reshape(predicted_future, (-1, hyper_params["future_length"], 2))
                predicted_list.append(predicted_future)

            gen_trajs = torch.DoubleTensor(np.transpose(predicted_list, (1,0,2,3))).to(device)
            tgt_trajs = torch.reshape(y, (-1, hyper_params["future_length"], 2 ))
            src_trajs = torch.reshape(x, (-1, hyper_params["past_length"], 2 ))

            rs_error3 = ((gen_trajs - tgt_trajs.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()

            num_agents = gen_trajs.size(0)
            num_agents3 = rs_error3.size(0)

            ade3 = rs_error3.mean(-1)
            fde3 = rs_error3[..., -1]

            minade3, _ = ade3.min(dim=-1)
            avgade3 = ade3.mean(dim=-1)
            minfde3, _ = fde3.min(dim=-1)
            avgfde3 = fde3.mean(dim=-1)

            batch_minade3 = minade3.mean()
            batch_minfde3 = minfde3.mean()
            batch_avgade3 = avgade3.mean()
            batch_avgfde3 = avgfde3.mean()

            epoch_minade3 += batch_minade3.item() * num_agents3
            epoch_avgade3 += batch_avgade3.item() * num_agents3
            epoch_minfde3 += batch_minfde3.item() * num_agents3
            epoch_avgfde3 += batch_avgfde3.item() * num_agents3

            epoch_agents += num_agents
            epoch_agents3 += num_agents3

            map_files = [map_file(args, sample_idx) for sample_idx in scene_id]
            num_tgt_trajs = torch.tensor(num_future_agents)
            num_src_trajs = torch.tensor(num_future_agents)

            cum_num_tgt_trajs = [0] + torch.cumsum(num_tgt_trajs, dim=0).tolist()
            cum_num_src_trajs = [0] + torch.cumsum(num_src_trajs, dim=0).tolist()

            src_trajs = src_trajs.cpu().numpy()

            tgt_trajs = tgt_trajs.cpu().numpy()

            gen_trajs = gen_trajs.cpu().numpy()

            batch_size = map_image.size(0)

            # print(cum_num_tgt_trajs)

            for i in range(batch_size):
                candidate_i = gen_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]
                tgt_traj_i = tgt_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]
                src_traj_i = src_trajs[cum_num_src_trajs[i]:cum_num_src_trajs[i + 1]]
                map_file_i = map_files[i]
                
                dao_i, dao_mask_i = dao(candidate_i, map_file_i, img=map_image[i])
                dac_i, dac_mask_i = dac(candidate_i, map_file_i, img=map_image[i])

                epoch_dao += dao_i.sum()
                dao_agents += dao_mask_i.sum()

                epoch_dac += dac_i.sum()
                dac_agents += dac_mask_i.sum()

                temp_min_sd, temp_max_sd, temp_std_sd, ast_agents = self_distance(candidate_i, tgt_traj_i, src_traj_i)
                epoch_min_sd = epoch_min_sd + temp_min_sd
                epoch_max_sd = epoch_max_sd + temp_max_sd
                epoch_std_sd = epoch_std_sd + temp_std_sd

                epoch_vo_angle += vo_angle(candidate_i, tgt_traj_i)

    return epoch_minade3, epoch_avgade3, epoch_minfde3, epoch_avgfde3, epoch_dao, epoch_dac, epoch_min_sd, epoch_max_sd, epoch_std_sd, epoch_vo_angle, epoch_agents3, epoch_agents, dao_agents, dac_agents

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    with open("./config/" + args.config_filename, 'r') as file:
        try:
            hyper_params = yaml.load(file, Loader = yaml.FullLoader)
        except:
            hyper_params = yaml.load(file)

    # 3) load dataset
    version = args.version
    data_type = args.data_type
    load_dir = args.load_dir
    min_angle = args.min_angle
    max_angle = args.max_angle
    print('min angle:', str(min_angle), ', max angle:', str(max_angle))



    val_dataset = DatasetQ10(version=version, load_dir=load_dir, data_partition='val',
                             shuffle=False, val_ratio=0.3, data_type='real', min_angle=min_angle,
                             max_angle=max_angle)


    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                              collate_fn=lambda x: nuscenes_pecnet_collate(x), num_workers=0)

    print(f'Valid Examples: {len(val_dataset)}')

    checkpoint = torch.load('./saved_models/{}'.format(args.load_file), map_location=device)
    hyper_params = checkpoint["hyper_params"]

    model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
    model = model.double().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])


    best_test_loss = 50 # start saving after this threshold
    best_endpoint_loss = 50
    N = hyper_params["n_values"]

    list_loss = []
    list_qloss = []
    list_ploss = []
    list_minade2, list_avgade2 = [], []
    list_minfde2, list_avgfde2 = [], []
    list_minade3, list_avgade3 = [], []
    list_minfde3, list_avgfde3 = [], []
    list_minmsd, list_avgmsd = [], []

    list_dao = []
    list_dac = []

    min_list_sd =[]
    max_list_sd =[]
    std_list_sd =[]
    list_angle = []

    # for e in tqdm(range(hyper_params['num_epochs']), desc='epoch'):
    for e in tqdm(range(1), desc='epoch'):
        epoch_minade3, epoch_avgade3, epoch_minfde3, epoch_avgfde3, \
        epoch_dao, epoch_dac, epoch_min_sd, epoch_max_sd, epoch_std_sd, epoch_vo_angle, \
        epoch_agents3, epoch_agents, dao_agents, dac_agents \
        = test_single_epoch(args, model, valid_loader, best_of_n = N, device=device, hyper_params=hyper_params)

        list_minade3.append(epoch_minade3 / epoch_agents3)
        list_avgade3.append(epoch_avgade3 / epoch_agents3)
        list_minfde3.append(epoch_minfde3 / epoch_agents3)
        list_avgfde3.append(epoch_avgfde3 / epoch_agents3)

        list_dao.append(epoch_dao / dao_agents)
        list_dac.append(epoch_dac / dac_agents)

        min_list_sd.append(epoch_min_sd / epoch_agents)
        max_list_sd.append(epoch_max_sd / epoch_agents)
        std_list_sd.append(epoch_std_sd / epoch_agents)
        list_angle.append(epoch_vo_angle / epoch_agents)

    test_minade3 = [np.mean(list_minade3), np.std(list_minade3)]
    test_avgade3 = [np.mean(list_avgade3), np.std(list_avgade3)]
    test_minfde3 = [np.mean(list_minfde3), np.std(list_minfde3)]
    test_avgfde3 = [np.mean(list_avgfde3), np.std(list_avgfde3)]

    test_dao = [np.mean(list_dao), np.std(list_dao)]
    test_dac = [np.mean(list_dac), np.std(list_dac)]

    test_ades = (test_minade3, test_avgade3)
    test_fdes = (test_minfde3, test_avgfde3)

    test_min_sd = [np.mean(min_list_sd), np.std(min_list_sd)]
    test_max_sd = [np.mean(max_list_sd), np.std(max_list_sd)]
    test_std_sd = [np.mean(std_list_sd), np.std(std_list_sd)]
    test_angle = [np.mean(list_angle), np.std(list_angle)]

    print("--Final Performane Report--")
    print("minADE3: {:.5f}±{:.5f}, minFDE3: {:.5f}±{:.5f}".format(test_minade3[0], test_minade3[1], test_minfde3[0], test_minfde3[1]))
    print("avgADE3: {:.5f}±{:.5f}, avgFDE3: {:.5f}±{:.5f}".format(test_avgade3[0], test_avgade3[1], test_avgfde3[0], test_avgfde3[1]))
    print("DAO: {:.5f}±{:.5f}, DAC: {:.5f}±{:.5f}".format(test_dao[0] * 10000.0, test_dao[1] * 10000.0, test_dac[0], test_dac[1]))
    print("minSD: {:.5f}±{:.5f}".format(test_min_sd[0], test_min_sd[1]))
    print("maxSD: {:.5f}±{:.5f}".format(test_max_sd[0], test_max_sd[1]))
    print("stdSD: {:.5f}±{:.5f}".format(test_std_sd[0], test_std_sd[1]))
    print("Angle: {:.5f}±{:.5f}".format(test_angle[0]*180/np.pi, test_angle[1]*180/np.pi))
    # with open(self.out_dir + '/metric.pkl', 'wb') as f:
    #     pkl.dump({"ADEs": test_ades,
    #                 "FDEs": test_fdes,
    #                 "DAO": test_dao,
    #                 "DAC": test_dac,
    #                 "minSD": test_min_sd,
    #                 "maxSD": test_max_sd,
    #                 "stdSD": test_std_sd,
    #                 "Angle": test_angle}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_dir', type=str, default='../datasets/nus_dataset')
    parser.add_argument('--data_type', type=str, default='real')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--min_angle', type=float, default=None)
    parser.add_argument('--max_angle', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=0, help="")
    parser.add_argument('--config_filename', '-cfn', type=str, default='optimal.yaml')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--load_file', '-lf', default="run7.pt")

    args = parser.parse_args()

    test(args)
