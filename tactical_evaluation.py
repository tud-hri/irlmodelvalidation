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
import glob
import multiprocessing as mp
import time

import numpy as np

from tacticalbehavior import TacticalBehavior
from processing.encryptiontools import load_encrypted_pickle, save_encrypted_pickle


def get_lane_id(y_position, upper_lane_markings, lower_lane_markings):
    all_markings = upper_lane_markings + lower_lane_markings
    all_markings.sort()
    all_markings = np.array(all_markings)
    lane_id = sum(y_position > all_markings)

    return lane_id


def check_for_collision(data_on_frame, agent_id, agent, agent_x, dataset, frame_index):
    for other_car_id in data_on_frame['id']:
        if other_car_id != agent_id:
            other_car_length = dataset.track_meta_data.at[other_car_id, 'width']
            other_car_width = dataset.track_meta_data.at[other_car_id, 'height']

            x_margin = agent.length / 2 + other_car_length / 2
            y_margin = agent.width / 2 + other_car_width / 2

            other_car_position = data_on_frame.loc[data_on_frame['id'] == other_car_id, ['x', 'y']].to_numpy()[0]
            other_car_position += np.array([other_car_length / 2, other_car_width / 2])
            distance_between_cars = abs(agent_x[frame_index, 0:2] - other_car_position)

            if distance_between_cars[0] < x_margin and distance_between_cars[1] < y_margin:
                return True, other_car_id
    return False, 0


def save_results(behavior_per_agent):
    total_agents = len(behavior_per_agent.keys())
    behavior_list = [result[0] for result in list(behavior_per_agent.values())]

    total_collisions = behavior_list.count(TacticalBehavior.COLLISION)
    total_off_road = behavior_list.count(TacticalBehavior.OFF_ROAD)
    total_lane_change = behavior_list.count(TacticalBehavior.LANE_CHANGE)
    total_car_following = behavior_list.count(TacticalBehavior.CAR_FOLLOWING)

    text = [
        'From the total number of %d agents, %d collided, %d went on an off-road adventure, and %d did a lane change. That means that %d did nothing ' \
        '(just car-following)' % (total_agents, total_collisions, total_off_road, total_lane_change, total_car_following)]

    text += ['']
    text += ['----------------------']
    text += ['| collisions     | %02d |' % total_collisions]
    text += ['| off-road       | %02d |' % total_off_road]
    text += ['| lane change    | %02d |' % total_lane_change]
    text += ['| car following  | %02d |' % total_car_following]
    text += ['----------------------']
    text += ['']

    text += ['Results for every agent:']
    text += ['---------------------------------------------------------------']
    text += ['| Dataset | Agent |    Behavior   |  Congested  |  Direction  |']
    text += ['---------------------------------------------------------------']

    for key, item in behavior_per_agent.items():
        text += ['|' + f'{key.split("_")[0]: ^9}' + '|' + f'{key.split("_")[1]: ^7}' + '|' + f'{str(item[0]): ^15}' + \
                 '|' + f'{item[1]: ^13}' + '|' + f'{item[2]: ^13}' + '|']
    text += ['---------------------------------------------------------------']

    with open('tactical_results.txt', 'w') as file:
        file.write("\n".join(text))


def evaluate_one_dataset(dataset_id_as_string, list_of_agent_files):
    dataset_id = int(dataset_id_as_string)
    dataset = load_encrypted_pickle('data/HighD/data/%02d.pkl' % dataset_id)
    behavior_per_agent = {}

    for filename in list_of_agent_files:
        simulation_dict = load_encrypted_pickle(filename)
        first_frame = simulation_dict['first_frame']
        last_frame = simulation_dict['last_frame']
        agent_x = simulation_dict['agent_x']
        agent = simulation_dict['agent']
        agent_id = simulation_dict['agent_id']

        agent_collided = False
        agent_went_off_road = False
        agent_did_lane_change = False
        agent_is_swerving = False

        try:
            agent.driving_direction
        except AttributeError:
            agent.driving_direction = 2

        agent_center_x = agent_x + np.array([agent.length / 2, agent.width / 2, 0, 0])

        if agent.driving_direction == 1:
            lane_markings = dataset.upper_lane_markings
        else:
            lane_markings = dataset.lower_lane_markings

        get_all_lane_ids = np.vectorize(lambda y: get_lane_id(y, dataset.upper_lane_markings, dataset.lower_lane_markings))
        all_lane_ids = get_all_lane_ids(agent_center_x[:, 1])
        number_of_lane_changes = sum(abs(all_lane_ids - np.roll(all_lane_ids, -1))[:-1])

        if number_of_lane_changes >= 1:
            print('agent %d in dataset %d did a lane change' % (agent_id, dataset_id))
            agent_did_lane_change = True

        if any(agent_center_x[:, 1] > lane_markings[-1]) or any(agent_center_x[:, 1] < lane_markings[0]):
            print('agent %d in dataset %d went off-road' % (agent_id, dataset_id))
            agent_went_off_road = True

        for frame_index, frame_number in enumerate(range(first_frame, last_frame + 1)):
            data_on_frame = dataset.track_data.loc[dataset.track_data['frame'] == frame_number, :]
            agent_collided, other_car_id = check_for_collision(data_on_frame, agent_id, agent, agent_center_x, dataset, frame_index)
            if agent_collided:
                print('agent %d in dataset %d collided with car %d' % (agent_id, dataset_id, other_car_id))
                break

        congested = 'Unknown'

        if agent.driving_direction == 1:
            driving_direction = 'West'
        else:
            driving_direction = 'East'

        if agent_collided:
            behavior = TacticalBehavior.COLLISION
        elif agent_went_off_road:
            behavior = TacticalBehavior.OFF_ROAD
        elif agent_did_lane_change:
            behavior = TacticalBehavior.LANE_CHANGE
        elif agent_is_swerving:
            behavior = TacticalBehavior.SWERVING
        else:
            behavior = TacticalBehavior.CAR_FOLLOWING

        results = (behavior, congested, driving_direction)
        simulation_dict['tactical_behavior'] = behavior
        simulation_dict['dataset_id'] = dataset_id
        save_encrypted_pickle(filename, simulation_dict)
        behavior_per_agent['%02d_%d' % (dataset_id, agent_id)] = results
    return behavior_per_agent


if __name__ == '__main__':
    os.chdir(os.getcwd() + '/..')
    t0 = time.time()
    all_agent_files = {}
    for data_id in range(1, 58):
        agent_files = glob.glob('data/HighD/data/%02d_agent_*_simulated.pkl' % data_id)
        if agent_files:
            all_agent_files[str(data_id)] = agent_files

    with mp.Pool(8) as p:
        results_list = p.starmap(evaluate_one_dataset, (all_agent_files.items()))

    total_results = {}

    for dataset_results_dict in results_list:
        total_results.update(dataset_results_dict)

    save_results(total_results)
    print('This took ' + str(time.time() - t0) + ' seconds')
