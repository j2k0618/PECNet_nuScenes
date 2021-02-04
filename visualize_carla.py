import os
import argparse

import torch
from torch.utils.data import DataLoader, ChainDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from pkyutils import DatasetQ10, nuscenes_collate, NusCustomParser
from nuscenes.prediction.input_representation.combinators import Rasterizer

import sys
sys.path.append("./utils/")
from models import *
from social_utils import *

import cv2
import natsort
import pickle

combinator = Rasterizer()

np.random.seed(777)
torch.manual_seed(777)

def log_determinant(sigma):
    det = sigma[:, :, 0, 0] * sigma[:, :, 1, 1] - sigma[:, :, 0, 1] ** 2
    logdet = torch.log(det + 1e-9)
    return logdet

'''for sanity check'''
def naive_social(p1_key, p2_key, all_data_dict):
	if abs(p1_key-p2_key)<4:
		return True
	else:
		return False

def find_min_time(t1, t2):
	'''given two time frame arrays, find then min dist (time)'''
	min_d = 9e4
	t1, t2 = t1[:8], t2[:8]

	for t in t2:
		if abs(t1[0]-t)<min_d:
			min_d = abs(t1[0]-t)

	for t in t1:
		if abs(t2[0]-t)<min_d:
			min_d = abs(t2[0]-t)

	return min_d

def find_min_dist(p1x, p1y, p2x, p2y):
	'''given two time frame arrays, find then min dist'''
	min_d = 9e4
	p1x, p1y = p1x[:8], p1y[:8]
	p2x, p2y = p2x[:8], p2y[:8]

	for i in range(len(p2x)):
		for j in range(len(p1x)):
			if ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5 < min_d:
				min_d = ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5

	return min_d

def social_and_temporal_filter(p1_key, p2_key, all_data_dict, time_thresh=48, dist_tresh=100):
	p1_traj, p2_traj = np.array(all_data_dict[p1_key]), np.array(all_data_dict[p2_key])
	p1_time, p2_time = p1_traj[:,1], p2_traj[:,1]
	p1_x, p2_x = p1_traj[:,2], p2_traj[:,2]
	p1_y, p2_y = p1_traj[:,3], p2_traj[:,3]

	if find_min_time(p1_time, p2_time)>time_thresh:
		return False
	if find_min_dist(p1_x, p1_y, p2_x, p2_y)>dist_tresh:
		return False

	return True

def mark_similar(mask, sim_list):
	# print(len(mask))
	# print(len(sim_list))
	for i in range(len(sim_list)):
		for j in range(len(sim_list)):
			mask[sim_list[i]][sim_list[j]] = 1

def initial_position(traj_batches):
	batches = []
	for b in traj_batches:
		starting_pos = b[:,4,:].copy()/1000 #starting pos is end of past, start of future. scaled down.
		batches.append(starting_pos)

	return batches

class Visualizer:
    def __init__(self, root='../datasets/nuscene/v1.0-mini', version='v1.0-mini'):
        self.root = root
        self.version = version
        self.sampling_time = 3
        self.agent_time = 0  # zero for static mask, non-zero for overlap
        self.layer_names = ['drivable_area', 'road_segment', 'road_block',
                       'lane', 'ped_crossing', 'walkway', 'stop_line',
                       'carpark_area', 'road_divider', 'lane_divider']
        self.colors = [(255, 255, 255), (100, 255, 255), (255, 100, 255),
                  (255, 255, 100), (100, 100, 255), (100, 255, 100), (255, 100, 100),
                  (100, 100, 100), (50, 100, 50), (200, 50, 50), ]

        self.dataset = NusCustomParser(
            root=self.root,
            version=self.version,
            sampling_time=self.sampling_time,
            agent_time=self.agent_time,
            layer_names=self.layer_names,
            colors=self.colors,
            resolution=0.1,
            meters_ahead=32,
            meters_behind=32,
            meters_left=32,
            meters_right=32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scene_channels = 3
        self.nfuture = int(3 * 2)
        self.crossmodal_attention = True
        self.beta = 0.1

        self.load_file = 'nusc_model_real0.6.pt'
        self.num_trajectories = 6

    def save_to_video(self, dataloader):
        results_idx = len(os.listdir('results'))
        results_dir = 'results/{}'.format(results_idx)
        os.mkdir(results_dir)
        print('save path: {}'.format(results_dir))

        # predict
        scene_ids, predicted, start = self.predict_path(dataloader)

        plt.figure(figsize=(10, 10))
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        for i, sample_idx in tqdm(enumerate(scene_ids), desc='plot'):
            # load data
            # print('############## sample_idx {}'.format(sample_idx))
            map_masks, map_img, agent_mask, xy_local, _, _, scene_id = self.dataset[sample_idx[0]]
            # agent_past, agent_future, agent_translation = xy_local
            combined_img = combinator.combine(np.append(map_masks[[0, 5, 8, 9]], agent_mask[np.newaxis, ...], axis=0))
            # visualize
            plt.title("Predicted")
            plt.imshow(combined_img, extent=[-32, 32, -32, 32], alpha=0.3)
            if len(xy_local[0]) != 0:
                self.draw_paths(plt.gca(), xy_local)
                plt.scatter(xy_local[2][:, 0], xy_local[2][:, 1], c='b', alpha=0.3)
            plt.xlim(-32, 32)
            plt.ylim(-32, 32)

            plt.scatter(start[i][:, 0], start[i][:, 1], color='g', s=22)

            for j in range(len(predicted[i])):
                paths = np.insert(predicted[i][j], 0, start[i][j], axis=1)
                for path in paths:
                    print(path)
                    plt.plot(path[:, 0], path[:, 1], color='r')

            # print(results_dir + '/{}.png'.format(i))
            plt.savefig(results_dir + '/{}.png'.format(i), dpi=150)
            plt.pause(0.001)
            plt.cla()
            #if i > 120:
            #    break

        # video_name = 'results/{}.avi'.format(results_idx)

        # images = [img for img in os.listdir(results_dir) if img.endswith(".png")]
        # images = natsort.natsorted(images)
        # if len(images)!=0:
        #     frame = cv2.imread(os.path.join(results_dir, images[5]))
        #     height, width, layers = frame.shape

        #     video = cv2.VideoWriter(video_name, 0, 2, (width, height))
        #     for image in tqdm(images, total=len(images), desc='video processing'):
        #         video.write(cv2.imread(os.path.join(results_dir, image)))
        #     cv2.destroyAllWindows()
        #     video.release()

    @staticmethod
    def draw_paths(ax, local_paths):
        past = local_paths[0]
        future = local_paths[1]
        translation = local_paths[2]
        for i in range(len(past)):
            if len(past[i]) != 0:
                path = np.append([translation[i]], past[i][-6:], axis=0)
                ax.plot(path[:, 0], path[:, 1], color='steelblue', linewidth=6, alpha=0.3)
            if len(future[i]) != 0:
                path = np.append([translation[i]], future[i][:6], axis=0)
                ax.plot(path[:, 0], path[:, 1], color='salmon', linewidth=6, alpha=0.3)

    def predict_path(self, data_pickle, batch_size=1):
        results_idx = []
        results_predicted = []
        results_pose = []
        device = self.device

        time_thresh = 0.6
        dist_tresh = 100000000000000

        checkpoint = torch.load('./saved_models/{}'.format(self.load_file), map_location=self.device)
        hyper_params = checkpoint["hyper_params"]
        hyper_params['sigma'] = 2.5

        N = self.num_trajectories #number of generated trajectories
        model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], False)
        model = model.double().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            for b, batch in tqdm(enumerate(data_pickle), total=len(data_pickle) // batch_size, desc='predict'):
                
                past_agents_traj, past_agents_traj_len, \
                future_agents_traj, future_agents_traj_len, future_agent_masks, \
                decode_start_pos, decode_start_vel,map_image, prior, scene_id = batch


                preprocessed = []
                for agent_id in range(len(past_agents_traj)):
                    past_len = past_agents_traj_len[agent_id]
                    future_len = future_agents_traj_len[agent_id]
                    if past_len + future_len == 10:
                        past = past_agents_traj[agent_id]
                        curr = np.array(decode_start_pos)[agent_id].reshape(-1,2)
                        future = future_agents_traj[agent_id]
                        path = np.concatenate((past, curr, future), axis=0)
                        frame_id = np.arange(0, len(path)).reshape(-1,1) * 0.5
                        agent_id_list = np.tile(np.array(agent_id), (len(path),1))
                        xy_coord = np.concatenate((frame_id, agent_id_list, path), axis=1)
                        preprocessed.append(xy_coord)



                ####################
                ### Preprocesing ###
                ####################
                mask_batch = [[0 for i in range(int(agent_num*1.5))] for j in range(int(agent_num*1.5))]
                full_dataset = []
                full_masks = []
                current_batch = []
                current_size = 0

                data_by_id = {}
                for idx in range(len(preprocessed)):
                    for frame_id, person_id, x, y in preprocessed[idx]:
                        if person_id not in data_by_id.keys():
                            data_by_id[person_id] = []
                        data_by_id[person_id].append([person_id, frame_id, x, y])

                all_data_dict = data_by_id.copy()

                while len(list(data_by_id.keys()))>0:
                    related_list = []
                    curr_keys = list(data_by_id.keys())

                    current_batch.append((all_data_dict[curr_keys[0]]))
                    related_list.append(current_size)
                    current_size+=1
                    del data_by_id[curr_keys[0]]

                    for i in range(1, len(curr_keys)):
                        if social_and_temporal_filter(curr_keys[0], curr_keys[i], all_data_dict, time_thresh, dist_tresh):
                            current_batch.append((all_data_dict[curr_keys[i]]))
                            related_list.append(current_size)
                            current_size+=1
                            del data_by_id[curr_keys[i]]

                    mark_similar(mask_batch, related_list)

                full_dataset.append(current_batch)
                mask_batch = np.array(mask_batch)
                full_masks.append(mask_batch[0:len(current_batch),0:len(current_batch)])

                ####################
                ####################
                ####################

                ###################################
                ### Social Dataset Consturction ###
                ###################################

                traj, masks = full_dataset, full_masks
                traj_new = []


                for t in traj:
                    t = np.array(t)
                    t = t[:,:,2:]
                    traj_new.append(t)



                masks_new = []
                for m in masks:
                    masks_new.append(m)

                traj_new = np.array(traj_new)
                # print(traj_new.shape)
                masks_new = np.array(masks_new)
                trajectory_batches = traj_new.copy()
                mask_batches = masks_new.copy()
                initial_pos_batches = np.array(initial_position(trajectory_batches)) #for relative positioning
                # print(initial_pos_batches)

                ###################################
                ###################################
                ###################################

                #################
                ### Inference ###
                #################

                traj = trajectory_batches[0]
                mask = mask_batches[0]
                initial_pos = initial_pos_batches[0]


                traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)

                x = traj[:, :hyper_params["past_length"], :]
                y = traj[:, hyper_params["past_length"]:, :]
                y = y.cpu().numpy()
                # reshape the data
                x = x.contiguous().view(-1, x.shape[1]*x.shape[2])
                x = x.to(device)

                future = y[:, :-1, :]
                dest = y[:, -1, :]
                all_l2_errors_dest = []
                all_guesses = []
                best_of_n = 20

                # print(np.reshape(x.cpu().numpy(), (-1, hyper_params["past_length"], 2))[0])
                # print(initial_pos.cpu().numpy()[0] *1000)

                for index in range(best_of_n):

                    dest_recon = model.forward(x, initial_pos, device=device)
                    dest_recon = dest_recon.cpu().numpy()
                    # print(dest_recon[0])
                    all_guesses.append(dest_recon)

                    l2error_sample = np.linalg.norm(dest_recon - dest, axis = 1)
                    all_l2_errors_dest.append(l2error_sample)

                all_l2_errors_dest = np.array(all_l2_errors_dest)
                all_guesses = np.array(all_guesses)
                # average error
                l2error_avg_dest = np.mean(all_l2_errors_dest)

                # choosing the best guess
                indices = np.argmin(all_l2_errors_dest, axis = 0)

                best_guess_dest = all_guesses[indices,np.arange(x.shape[0]),  :]

                # taking the minimum error out of all guess
                l2error_dest = np.mean(np.min(all_l2_errors_dest, axis = 0))

                # back to torch land
                best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)

                predicted_list = []
                for goal in all_guesses:
                    # using the best guess for interpolation
                    goal = torch.DoubleTensor(goal).to(device)
                    interpolated_future = model.predict(x, goal, mask, initial_pos)
                    interpolated_future = interpolated_future.cpu().numpy()
                    goal = goal.cpu().numpy()
                    # final overall prediction
                    predicted_future = np.concatenate((interpolated_future, goal), axis = 1)
                    predicted_future = np.reshape(predicted_future, (-1, hyper_params["future_length"], 2))
                                        #  - torch.unsqueeze(torch.tensor(initial_pos * 1000), 1).cpu().numpy()
                    predicted_list.append(predicted_future)

                #################
                #################
                #################


                predicted_list = np.transpose(predicted_list, (1,0,2,3))

                # interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
                # interpolated_future = interpolated_future.cpu().numpy()
                # best_guess_dest = best_guess_dest.cpu().numpy()
                # best_predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis = 1)
                # best_predicted_future = np.reshape(best_predicted_future, (-1, hyper_params["future_length"], 2))
                # l2error_overall = np.mean(np.linalg.norm(y - best_predicted_future, axis = 2))

                # print(l2error_overall)
                # print(l2error_dest)
                results_idx.append(scene_id)
                results_predicted.append(predicted_list)
                # results_predicted.append(torch.unsqueeze(torch.tensor(y), 1).cpu().numpy())
                results_pose.append(torch.squeeze(torch.tensor(initial_pos * 1000), 1).cpu().numpy())

        return results_idx, results_predicted, results_pose


def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nfuture = int(3 * 2)

    version = args.version
    data_type = args.data_type
    load_dir = args.load_dir
    min_angle = args.min_angle
    max_angle = args.max_angle
    print('min angle:', str(min_angle), ', max angle:', str(max_angle))

    # dataset = DatasetQ10(version=version, load_dir=load_dir, data_partition='val',
    #                      shuffle=False, val_ratio=0.3, data_type=data_type, min_angle=min_angle, max_angle=max_angle)
    # data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
    #                          collate_fn=lambda x: nuscenes_collate(x), num_workers=args.num_workers)
    
    with open('/home/user/jegyeong/datf/cache/carla_val_please_1496.pickle.pickle', 'rb') as f:
        data_pickle = pickle.load(f)

    # print(f'Test Examples: {len(dataset)}')


    viz = Visualizer(root='{}/original_small/{}'.format(load_dir, version), version=version)
    viz.save_to_video(data_pickle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default='../datasets/nus_dataset')
    parser.add_argument('--data_type', type=str, default='real')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--min_angle', type=float, default=None)
    parser.add_argument('--max_angle', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=1, help="")


    args = parser.parse_args()

    visualize(args)
