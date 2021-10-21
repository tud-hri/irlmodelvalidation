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
import os
import time

import numpy as np

from dataobjects import HighDDataset
from dataobjects.enums import HighDDatasetID
from irlmodelvalidation import irlagenttools
from irlmodelvalidation.irlagent import IRLAgent
from processing.encryptiontools import save_encrypted_pickle


def run_agent(theta, data: HighDDataset, agent_id, c, sigma_x, sigma_y, save_as_grid_search_file=False, path_to_data_folder='data\\', verbose=True):
    t0 = time.time()

    # road parameters
    planner_dt = 1 / data.frame_rate
    sim_dt = 1 / data.frame_rate
    N = 5

    road_boundaries = np.array([data.upper_lane_markings[0], data.upper_lane_markings[-1], data.lower_lane_markings[0], data.lower_lane_markings[-1]])
    lane_centers = []

    for index in range(len(data.upper_lane_markings) - 1):
        lane_centers.append((data.upper_lane_markings[index + 1] - data.upper_lane_markings[index]) / 2 + data.upper_lane_markings[index])

    for index in range(len(data.lower_lane_markings) - 1):
        lane_centers.append((data.lower_lane_markings[index + 1] - data.lower_lane_markings[index]) / 2 + data.lower_lane_markings[index])

    average_lane_width = abs(road_boundaries[1] - road_boundaries[0]) / (len(lane_centers) / 2)

    road_boundaries += np.array([-average_lane_width / 2, average_lane_width / 2, -average_lane_width / 2, average_lane_width / 2])

    # agent parameters
    agent_length = data.track_meta_data.at[agent_id, 'width']
    agent_width = data.track_meta_data.at[agent_id, 'height']

    desired_velocity = data.track_data.loc[data.track_data['id'] == agent_id, 'xVelocity'].abs().max()

    first_frame = data.track_data.loc[data.track_data['id'] == agent_id, 'frame'].min()
    last_frame = data.track_data.loc[data.track_data['id'] == agent_id, 'frame'].max()

    driving_direction = data.track_meta_data.at[agent_id, 'drivingDirection']
    # create agent
    agent = IRLAgent(agent_width, agent_length, theta, road_boundaries, lane_centers,
                           desired_velocity, driving_direction, planner_dt, N, c=c, sigma_x=sigma_x, sigma_y=sigma_y)

    starting_point_data = data.track_data.loc[(data.track_data['frame'] == first_frame) & (data.track_data['id'] == agent_id), :]
    x0 = starting_point_data[['x', 'y', 'xVelocity', 'yVelocity']].iloc[0, :].to_numpy()

    agent_x = np.array([[0.0, 0.0, 0.0, 0.0]] * (last_frame - first_frame + 1))
    agent_x[0, :] = x0

    # simulate agent
    for frame_index, frame in enumerate(range(first_frame, last_frame)):
        horizon_data = data.track_data.loc[data.track_data['frame'].isin([x for x in range(frame, frame + N)]), :]

        surrounding_car_sizes, surrounding_cars_positions = irlagenttools.get_surrounding_cars_information(horizon_data, data.track_meta_data, frame, frame + N,
                                                                                                      agent_id)
        x_for_agent = agent_x[frame_index, :] + np.array([agent_length / 2, agent_width / 2, 0.0, 0.0])  # convert from top left corner to center coordinates

        u = agent.calculate_action(x_for_agent, surrounding_cars_positions, surrounding_car_sizes)
        agent_x[frame_index + 1, :] = agent.calculate_new_state(agent_x[frame_index, :], u, sim_dt)

    t1 = time.time()
    if verbose:
        print('Simulating for this agent took %0.3f seconds' % (t1 - t0))

    save_dict = {'first_frame': first_frame,
                 'last_frame': last_frame,
                 'agent': agent,
                 'agent_x': agent_x,
                 'agent_id': agent_id}

    if save_as_grid_search_file:
        save_encrypted_pickle(path_to_data_folder + 'grid_search/agent_%d_c_%.2f_x_%.2f_y_%.2f.pkl' % (agent_id, agent.C2, agent.sigma_x, agent.sigma_y),
                              save_dict)
    else:
        save_encrypted_pickle(path_to_data_folder + '%02d_agent_%d_simulated.pkl' % (data.recording_id, agent_id), save_dict)


if __name__ == '__main__':
    os.chdir(os.getcwd() + '/..')
    dataset_id = HighDDatasetID.DATASET_01
    # load data
    data = HighDDataset.load(dataset_id)

    run_agent(theta=np.array([-4.25053061e+00, 7.41977117e+00, -3.72867737e+00, -7.65764053e+03]), data=data, agent_id=60)
