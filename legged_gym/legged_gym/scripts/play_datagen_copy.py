# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import export_policy_as_jit_actor,export_policy_as_jit_encoder,class_to_dict

import numpy as np
import torch
import pickle
import random

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    class_to_dict(env_cfg)
    class_to_dict(train_cfg)
    
    # with open('env_cfg.pkl', 'wb') as f:
    #     pickle.dump(class_to_dict(env_cfg), f)
    # with open('train_cfg.pkl', 'wb') as f:
    #     pickle.dump(train_cfg, f)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs,obs_hist = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit_actor(ppo_runner.alg.actor_critic, path)
        export_policy_as_jit_encoder(ppo_runner.alg.actor_critic,path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 1000 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # Defining velocity ranges for vx, vy and wz
    num_commands = 4 # 50
    num_env_steps = 499
    x_vel_range = np.linspace(-1., 1., 9)
    x_vel_range = np.delete(x_vel_range, 4)
    y_vels_range = np.linspace(-0.6, 0.6, 5)
    y_vels_range = np.delete(y_vels_range, 2)
    w_range = np.linspace(-1., 1., 9)
    commands = torch.zeros_like(env.commands)

    #defining input and output lists for writing into json

    inputs = []
    targets = []

    # Enter episodes : num commands
    for j in range(env.num_envs):

        val = random.randint(1, 3)
        print(val)
        if val == 1:
            (x_vel_cmd, y_vel_cmd, yaw_vel_cmd) = (random.choice(x_vel_range), 0., 0.)

        elif val == 2:
            (x_vel_cmd, y_vel_cmd, yaw_vel_cmd) = (0., random.choice(y_vels_range), 0.)

        else:
            (x_vel_cmd, y_vel_cmd, yaw_vel_cmd) = (0., 0., random.choice(w_range))

        (x_vel_cmd, y_vel_cmd, yaw_vel_cmd) = (0., 0., 0.25)

        commands[j, 0] = x_vel_cmd
        commands[j, 1] = y_vel_cmd
        commands[j, 2] = yaw_vel_cmd

    env.commands[:, :] = commands[:, :]

        # print(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)
        # (x_vel_cmd, y_vel_cmd, yaw_vel_cmd) = (0., 0., random.choice(w_range))
        # env.commands[:, 0] = x_vel_cmd
        # env.commands[:, 1] = y_vel_cmd
        # env.commands[:, 2] = yaw_vel_cmd

        # print(env.commands[:, :])

        # obs, obs_hist = env.get_observations()
        # print(obs[:, 6:9])
        # print(obs[:, 6:9])
        # print(env.commands[:, :3])
        # print("chosen velocities vx, vy, wz:", x_vel_cmd, y_vel_cmd, yaw_vel_cmd)
        # print("iteration:", i)
    for j in range(num_env_steps):
        obs_hist_reshaped = obs_hist.view((env.num_envs, 5, 45))
        base_ang_vel = obs_hist_reshaped[:, :, :3]
        proj_gravity = obs_hist_reshaped[:, :, 3:6]
        vel_commands = env.commands[:, :3]
        print(vel_commands)
        dof_pos = obs_hist_reshaped[:, :, 9:21]
        dof_vel = obs_hist_reshaped[:, :, 21:33]
        prev_actions = obs_hist_reshaped[:, :, 33:45]

        diff_obs = torch.cat((base_ang_vel, proj_gravity, dof_pos, dof_vel, prev_actions), dim=2).view(env.num_envs, 210)
        diff_obs = torch.cat((vel_commands, diff_obs), dim=1)

        with torch.no_grad():
            actions = policy(obs.detach(), obs_hist.detach())
            obs, _, _, obs_hist, rews, dones, infos = env.step(actions.detach())

        inputs.append(diff_obs.cpu().numpy())
        targets.append(actions.cpu().numpy())


    y_vel_cmd = 0.0
    yaw_vel_cmd =0.0
    num_eval_steps = 5000
    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) 
    joint_positions = np.zeros((num_eval_steps, 12))
    root_positions = np.zeros((num_eval_steps, 13))
    policy_actions = np.zeros((num_eval_steps, 12))
    gravity_vector = np.zeros((num_eval_steps, 3))
    dof_pos_change = np.zeros((num_eval_steps, 12))
    commands = np.zeros((num_eval_steps, 3))
    commanded_vel = np.zeros((num_eval_steps, 3))
    dof_vel = np.zeros((num_eval_steps, 12))
    obs_hist_buf = torch.zeros(1, 150, dtype=torch.float)
    obs_hist_diff = np.zeros((num_eval_steps,150))
    base_lin_vel = np.zeros((num_eval_steps,3))
    base_ang_vel =np.zeros((num_eval_steps, 3))
    # obs_val = env.reset()
    # print(obs_val)

    for i in range(num_eval_steps):
        
        
        if i%100==0:
            # env.reset_idx([0])
            # obs_val = env.reset()
            # observations, observations_hist = env.get_observations()
            # print(observations_hist.shape)

            x_vel_cmd =  random.sample(vel_range,1)

        env.commands[:, 0] = x_vel_cmd[0]
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        # obs_hist_diff[i] = obs_hist_buf.cpu().numpy()
        target_x_vels[i] = x_vel_cmd[0]
        base_lin_vel[i] =  env.base_lin_vel[0,:].cpu()
        base_ang_vel[i] = env.base_ang_vel[0,:].cpu()
        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()
        root_positions[i] = env.root_states[0,:].cpu()
        gravity_vector[i] = env.projected_gravity[0,:].cpu()
        commands[i] = env.commands[0,:3].cpu()
        commanded_vel[i] = env.commands[0,:3].cpu()
        dof_pos_change[i] = env.dof_pos.cpu()[0:,] - env.default_dof_pos.cpu()[0,:]
        dof_vel[i] = env.dof_vel[0:,:].cpu() 
        obs_hist_buf = obs_hist_buf[:,30:]
        obs_hist_buf = torch.cat((obs_hist_buf,obs[:,3:33].cpu()),dim = -1)
        obs_hist_diff[i] = obs_hist_buf.cpu().numpy()
        # import ipdb;ipdb.set_trace()
        with torch.no_grad():
            actions = policy(obs.detach(),obs_hist.detach())
            obs, _, _, obs_hist, rews, dones, infos = env.step(actions.detach())
        policy_actions[i] = actions.cpu()
        
    

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
     
    data ={}
    data["commanded_vel"] = commanded_vel
    data["policy_actions"] = policy_actions
    data ["gravity_vector"] = gravity_vector
    data["commands"] = commands
    data["dof_pos_change"] = dof_pos_change
    data["dof_vel"] = dof_vel
    data ["base_lin_vel"] = base_lin_vel
    data ["base_ang_vel"] = base_ang_vel
    data ["obs_hist_diff"] = obs_hist_diff

    pickle.dump(data, open('dq_cv_x_0.1_1.pkl', 'wb'))
    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), policy_actions, linestyle="-", label="Measured")
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.show()
        # if i < stop_state_log:
        #     logger.log_states(
        #         {
        #             'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
        #             'dof_pos': env.dof_pos[robot_index, joint_index].item(),
        #             'dof_vel': env.dof_vel[robot_index, joint_index].item(),
        #             'dof_torque': env.torques[robot_index, joint_index].item(),
        #             'command_x': env.commands[robot_index, 0].item(),
        #             'command_y': env.commands[robot_index, 1].item(),
        #             'command_yaw': env.commands[robot_index, 2].item(),
        #             'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
        #             'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
        #             'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
        #             'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
        #             'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
        #             'dof_pos_0': env.dof_pos[robot_index, 0].item(),
        #             'dof_pos_1': env.dof_pos[robot_index, 1].item(),
        #             'dof_pos_2': env.dof_pos[robot_index, 2].item(),
        #             'dof_pos_3': env.dof_pos[robot_index, 3].item(),
        #             'dof_pos_4': env.dof_pos[robot_index, 4].item(),
        #             'dof_pos_5': env.dof_pos[robot_index, 5].item(),
        #             'dof_pos_6': env.dof_pos[robot_index, 6].item(),
        #             'dof_pos_7': env.dof_pos[robot_index, 7].item(),
        #             'dof_pos_8': env.dof_pos[robot_index, 8].item(),
        #             'dof_pos_9': env.dof_pos[robot_index, 9].item(),
        #             'dof_pos_10': env.dof_pos[robot_index, 10].item(),
        #             'dof_pos_11': env.dof_pos[robot_index, 11].item(),
                    
        #         }
        #     )
        # elif i==stop_state_log:
        #     logger.plot_states()
        # if  0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes>0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i==stop_rew_log:
        #     logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
