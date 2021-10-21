"""
Copyright 2021, Olger Siebinga (o.siebinga@tudelft.nl)

This file is part of the module irlmodelvalidation.

irlmodelvalidation is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

irlmodelvalidation is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with irlmodelvalidation.  If not, see <https://www.gnu.org/licenses/>.
"""
import math
import os

import multiprocessing as mp
import autograd.numpy as np
from autograd import elementwise_grad
from scipy import optimize

from irlmodelvalidation.irlagent import IRLAgent
from dataobjects import HighDDataset
from dataobjects.enums import HighDDatasetID
from irlmodelvalidation.rewardgradients import calculate_g_and_H_autograd
from irlmodelvalidation.evaluatemodel import run_agent
from irlmodelvalidation.irlagenttools import get_surrounding_cars_information
from processing.encryptiontools import load_encrypted_pickle, save_encrypted_pickle


def fit_theta_and_simulate(agent_id, dataset_index, road_boundaries, lane_centers, planner_dt, sim_dt, N, c=0.18, sigma_x=10.0, sigma_y=1.4, grid_search=False,
                           verbose=True, path_to_data_folder='data\\'):
    if verbose:
        print('Training on vehicle %d in dataset %02d, with PID %d' % (agent_id, dataset_index, os.getpid()))

    data = load_encrypted_pickle(path_to_data_folder + '%02d.pkl' % dataset_index)

    first_frame = data.track_data.loc[data.track_data['id'] == agent_id, 'frame'].min()
    last_frame = data.track_data.loc[data.track_data['id'] == agent_id, 'frame'].max()

    number_of_demonstrations = math.floor((last_frame - first_frame) / N)
    desired_velocity = data.track_data.loc[data.track_data['id'] == agent_id, 'xVelocity'].abs().max()

    u_h_values = np.array([[0.0] * N * 2] * number_of_demonstrations)
    x0 = np.array([[0.0] * 2] * number_of_demonstrations)
    x_r_values = []

    # agent parameters
    agent_length = data.track_meta_data.at[agent_id, 'width']
    agent_width = data.track_meta_data.at[agent_id, 'height']
    driving_direction = data.track_meta_data.at[agent_id, 'drivingDirection']

    for demonstration_number in range(number_of_demonstrations):
        frames = [f for f in range(first_frame + demonstration_number * N, first_frame + (demonstration_number + 1) * N)]
        frames_for_opponent = [f for f in range(first_frame + demonstration_number * N, first_frame + 1 + (demonstration_number + 1) * N)]

        state_index = demonstration_number
        u_h_demo = data.track_data.loc[(data.track_data['id'] == agent_id) & data.track_data['frame'].isin(frames), ['xVelocity', 'yVelocity']]
        u_h_values[state_index] = u_h_demo.to_numpy().flatten()

        horizon_data = data.track_data.loc[data.track_data['frame'].isin(frames_for_opponent), :]
        _, surrounding_cars_positions = get_surrounding_cars_information(horizon_data, data.track_meta_data, frames_for_opponent[0], frames_for_opponent[-1],
                                                                         agent_id)

        if surrounding_cars_positions.any():
            flat_surrounding_car_positions = surrounding_cars_positions.reshape(surrounding_cars_positions.shape[0],
                                                                                surrounding_cars_positions.shape[1] * surrounding_cars_positions.shape[2])
            flat_surrounding_car_positions = list(flat_surrounding_car_positions)
            x_r_values.append(flat_surrounding_car_positions)
        else:
            x_r_values.append([])

        first_frame_x_h = data.track_data.loc[(data.track_data['id'] == agent_id) & (data.track_data['frame'] == frames[0]), ['x', 'y']]
        x0[state_index] = first_frame_x_h.to_numpy()[0]

        # convert from top left corner to center coordinates
        x0[state_index] += np.array([agent_length / 2, agent_width / 2])

    # create agent
    agent = IRLAgent(0.0, 0.0, np.array([4.0, 0.02, 0.02, 1.5]), road_boundaries, lane_centers, desired_velocity, driving_direction, planner_dt, N, c=c,
                           sigma_x=sigma_x,
                           sigma_y=sigma_y)

    constants_values = (agent.C2, desired_velocity, road_boundaries, lane_centers, agent.sigma_x, agent.sigma_y, driving_direction)

    if verbose:
        print('determine H and g')
    g, H = calculate_g_and_H_autograd(constants_values, sim_dt, N)

    def likelihood(theta_values):
        l_total = 0.0
        successful_evaluations = 0

        for demonstration_number in range(number_of_demonstrations):

            g_values = g(u_h_values[demonstration_number], theta_values, x_r_values[demonstration_number], x0[demonstration_number])
            H_values = H(u_h_values[demonstration_number], theta_values, x_r_values[demonstration_number], x0[demonstration_number])

            H_values -= np.eye(N * 2) * 1e-1

            H_inv = np.linalg.inv(H_values)
            H_det = np.linalg.det(-H_values)

            l = np.dot(np.dot(g_values, H_inv), g_values) + np.log(H_det)
            if not np.isnan(l):
                l_total += l
                successful_evaluations += 1
        return -l_total / successful_evaluations

    if verbose:
        print('starting optimization')

    jac = elementwise_grad(likelihood)
    try:
        result = optimize.minimize(likelihood, np.array([-3.0, 5.0, -0.10, -2000.0]), jac=jac, options={'disp': True}, method='bfgs')
        if verbose:
            print('Theta for agent %d = ' % agent_id + str(result.x))

        run_agent(theta=result.x, data=data, agent_id=agent_id, c=agent.C2, sigma_x=agent.sigma_x, sigma_y=agent.sigma_y, save_as_grid_search_file=grid_search,
                  path_to_data_folder=path_to_data_folder, verbose=verbose)
    except Exception as e:
        if verbose:
            print("WARNING: fitting or running for agent %d failed with the exception below, now continuing with next agent" % agent_id)
            print(e)


def detect_single_lane_changes(data: HighDDataset):
    interesting_lane_changes = []
    cars_with_one_lane_change = data.track_meta_data.loc[data.track_meta_data['numLaneChanges'] == 1].index

    for car in cars_with_one_lane_change:
        initial_lane = data.track_data.loc[data.track_data['id'] == car, 'laneId'].iat[0]
        final_lane = data.track_data.loc[data.track_data['id'] == car, 'laneId'].iat[-1]
        driving_direction = data.track_meta_data.at[car, 'drivingDirection']

        if driving_direction == 1 and (final_lane - initial_lane) > 0:
            interesting_lane_changes.append(car)
        elif driving_direction == 2 and (final_lane - initial_lane) < 0:
            interesting_lane_changes.append(car)

    return interesting_lane_changes


if __name__ == '__main__':
    os.chdir(os.getcwd() + '/..')
    dataset_id = HighDDatasetID.DATASET_01
    dataset_index = dataset_id.value

    data = HighDDataset.load(dataset_id)

    # road parameters
    planner_dt = 1 / data.frame_rate
    sim_dt = 1 / data.frame_rate
    N = 5
    c = 0.14
    sigma_x = 15.0
    sigma_y = 1.4
    path_to_data_folder = 'data/HighD/data/'
    verbose = True
    is_grid_search = False
    workers = 4

    road_boundaries = np.array([data.upper_lane_markings[0], data.upper_lane_markings[-1], data.lower_lane_markings[0], data.lower_lane_markings[-1]])
    lane_centers = []

    for index in range(len(data.upper_lane_markings) - 1):
        lane_centers.append((data.upper_lane_markings[index + 1] - data.upper_lane_markings[index]) / 2 + data.upper_lane_markings[index])

    for index in range(len(data.lower_lane_markings) - 1):
        lane_centers.append((data.lower_lane_markings[index + 1] - data.lower_lane_markings[index]) / 2 + data.lower_lane_markings[index])

    average_lane_width = abs(road_boundaries[1] - road_boundaries[0]) / len(lane_centers)

    road_boundaries += np.array([-average_lane_width / 2, average_lane_width / 2, -average_lane_width / 2, average_lane_width / 2])

    # demonstrations to use
    ego_ids = detect_single_lane_changes(data)
    ego_ids = ego_ids[0:4]

    args = zip(ego_ids, [dataset_index] * len(ego_ids), [road_boundaries] * len(ego_ids), [lane_centers] * len(ego_ids), [planner_dt] * len(ego_ids),
               [sim_dt] * len(ego_ids), [N] * len(ego_ids), [c] * len(ego_ids), [sigma_x] * len(ego_ids), [sigma_y] * len(ego_ids),
               [is_grid_search] * len(ego_ids), [verbose] * len(ego_ids), [path_to_data_folder] * len(ego_ids))

    if verbose:
        print("Starting training with %d workers" % workers)

    with mp.Pool(workers) as p:
        p.starmap(fit_theta_and_simulate, args)

    if verbose:
        print('Dataset %02d done' % dataset_index)
