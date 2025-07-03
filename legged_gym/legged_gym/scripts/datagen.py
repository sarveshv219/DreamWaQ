import json

import torch
import os
import numpy as np

folder_pth = r"/home/narayanan/PycharmProjects/diffusion_datasets/go1/dwaq"
fn1 = "dwaq_flat.json"
fn2 = "dwaq_uneven.json"
fn3 = "dwaq_stairs.json"
fn4 = "dwaq_slopes.json"
fn5 = "dwaq_hist_slopes.json"


class Datagen(object):
    def __init__(self):
        return

    def process_data(self, data):
        datapoints1 = []
        datapoints2 = []
        for d in data:
            datapoint1 = self.datagen_single_action(d[0])
            datapoint2 = self.datagen_horizon_action(d[0])

            tar2_shape = np.array(datapoint2['targets']).shape
            # print(tar2_shape)
            zero2_list = np.zeros(tar2_shape).tolist()
            val = zero2_list[0]
            # print(zero2_list)

            if val in datapoint1['targets']:
                pass

            else:
                datapoints1.append(datapoint1)

            if val in datapoint2['targets']:
                pass

            else:
                datapoints2.append(datapoint2)

        return datapoints1, datapoints2

    def datagen_single_action(self, obs):
        obs_history = np.concatenate((obs[-5:-1, 3:6], obs[-5:-1, :3], obs[-5:-1, 18:-4]),
                                axis=1)  # joint pos(12) + joint vels(12) + gravity_vec(3) + vel commands(3) + at-1(12) + at-2(12)
        targets = obs[-1, -24:-12]  # actions at t-1
        data_point = {
            "inputs": obs_history.tolist(),
            "targets": targets.tolist(),
        }

        return data_point

    # def get_inputs_wtw(self, obs):
    #     # obs_history = torch.cat((obs[-5:-1, 3:6], obs[-5:-1, :3], obs[-5:-1, 18:-4]),
    #     #                         dim=1)
    #     obs_history = torch.cat((obs[-1, 3:6], obs[-1, :3], obs[-1, 18:-4]),
    #                             dim=-1).unsqueeze(0)
    #
    #     obs_history = obs[-1, -24:].unsqueeze(0)
    #     # print(obs_history.shape)
    #
    #     return obs_history

    def datagen_horizon_action(self, obs) :
        obs_history = np.concatenate((obs[-8:-4, 3:6], obs[-8:-4, :3], obs[-8:-4, 18:-4]),
                                axis=1)  # joint pos(12) + joint vels(12) + gravity_vec(3) + vel commands(3) + at-1(12) + at-2(12)
        targets = obs[-4:, -24:12]  # actions at t-1
        data_point = {
            "inputs": obs_history.tolist(),
            "targets": targets.tolist(),
        }

        return data_point
    #
    # def write_into_json(self, data):
    #     data1, data2 = self.process_data(data)
    #     file_pth1 = os.path.join(folder_pth, fn1)
    #     file_pth_2 = os.path.join(folder_pth, fn2)
    #     with open(file_pth1, 'w') as f:
    #         json.dump(data1, fp=f, indent=4)
    #
    #     with open(file_pth_2, 'w') as f:
    #         json.dump(data2, fp=f, indent=4)

    def write_into_json_base(self, data, actions):
        # print("inp shape-----------------", data.shape)
        data = data.tolist()
        actions = actions.tolist()
        # print(len(actions))
        dataset = []
        for obs, act in zip(data, actions):
            # print(obs)
            datapoint = {
                "inputs": obs,
                "targets": act,
            }
            dataset.append(datapoint)

        file_pth = os.path.join(folder_pth, fn5)
        with open(file_pth, 'w') as f:
            json.dump(dataset, fp=f, indent=4)









