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
import glob
import os

import numpy as np

from tactical_evaluation import get_lane_id, check_for_collision
from processing.encryptiontools import load_encrypted_pickle, save_encrypted_pickle
from .tacticalbehavior import TacticalBehavior


def evaluate(dataset, list_of_agent_files):
    behavior_per_agent = {}

    for filename in list_of_agent_files:
        simulation_dict = load_encrypted_pickle(filename)
        first_frame = simulation_dict['first_frame']
        last_frame = simulation_dict['last_frame']
        agent_x = simulation_dict['agent_x']
        agent = simulation_dict['agent']
        agent_id = simulation_dict['agent_id']

        parameters_as_string = 'c_%.2f_x_%.2f_y_%.2f' % (agent.C2, agent.sigma_x, agent.sigma_y)

        if parameters_as_string not in behavior_per_agent.keys():
            behavior_per_agent[parameters_as_string] = {}

        agent_collided = False
        agent_went_off_road = False
        agent_did_lane_change = False

        try:
            agent.driving_direction
        except AttributeError:
            agent.driving_direction = 2

        agent_center_x = agent_x + np.array([agent.length / 2, agent.width / 2, 0, 0])

        if agent.driving_direction == 1:
            lane_markings = dataset.upper_lane_markings
        else:
            lane_markings = dataset.lower_lane_markings

        last_lane_id = get_lane_id(agent_center_x[0, 1], dataset.upper_lane_markings, dataset.lower_lane_markings)

        for frame_index, frame_number in enumerate(range(first_frame, last_frame + 1)):
            if agent_center_x[frame_index, 1] > lane_markings[-1] or agent_center_x[frame_index, 1] < lane_markings[0]:
                agent_went_off_road = True
                break

            current_lane_id = get_lane_id(agent_center_x[frame_index, 1], dataset.upper_lane_markings, dataset.lower_lane_markings)
            if last_lane_id != current_lane_id:
                last_lane_id = current_lane_id
                agent_did_lane_change = True

            data_on_frame = dataset.track_data.loc[dataset.track_data['frame'] == frame_number, :]
            agent_collided, other_car_id = check_for_collision(data_on_frame, agent_id, agent, agent_center_x, dataset, frame_index)
            if agent_collided:
                break

        if agent_collided:
            behavior = TacticalBehavior.COLLISION
        elif agent_went_off_road:
            behavior = TacticalBehavior.OFF_ROAD
        elif agent_did_lane_change:
            behavior = TacticalBehavior.LANE_CHANGE
        else:
            behavior = TacticalBehavior.CAR_FOLLOWING

        behavior_per_agent[parameters_as_string]['agent_%d_' % agent_id] = behavior
    return behavior_per_agent


def save_results_as_text(behavior_per_agent):

    text = []

    for parameter_set, behavior_dict in behavior_per_agent.items():
        total_agents = len(behavior_dict.keys())
        behavior_list = list(behavior_dict.values())

        total_collisions = behavior_list.count(TacticalBehavior.COLLISION)
        total_off_road = behavior_list.count(TacticalBehavior.OFF_ROAD)
        total_lane_change = behavior_list.count(TacticalBehavior.LANE_CHANGE)
        total_car_following = behavior_list.count(TacticalBehavior.CAR_FOLLOWING)

        text += ['Results for ' + parameter_set]
        text += ['Total succeeded = ' + str(total_agents)]
        text += ['']
        text += ['----------------------']
        text += ['| collisions     | %02d |' % total_collisions]
        text += ['| off-road       | %02d |' % total_off_road]
        text += ['| lane change    | %02d |' % total_lane_change]
        text += ['| car following  | %02d |' % total_car_following]
        text += ['----------------------']
        text += ['']

        text += ['---------------------------------------------------------------']

    with open('grid_search_results.txt', 'w') as file:
        file.write("\n".join(text))


if __name__ == '__main__':
    os.chdir(os.getcwd() + '\\..')

    dataset_id = 1

    results = load_encrypted_pickle('data/grid_search/results_%02d.pkl' % dataset_id)

    if results is None:
        data = load_encrypted_pickle('data/%02d.pkl' % dataset_id)
        agent_files = glob.glob('data/grid_search/agent_*.pkl')

        results = evaluate(dataset=data, list_of_agent_files=agent_files)

        save_encrypted_pickle('data/grid_search/results_%02d.pkl' % dataset_id, results)
        save_results_as_text(results)

    total_collisions = {}
    total_off_road = {}
    total_lane_change = {}
    total_car_following = {}

    for parameter_set, behavior_dict in results.items():
        total_agents = len(behavior_dict.keys())
        behavior_list = list(behavior_dict.values())

        total_collisions[parameter_set] = behavior_list.count(TacticalBehavior.COLLISION)
        total_off_road[parameter_set] = behavior_list.count(TacticalBehavior.OFF_ROAD)
        total_lane_change[parameter_set] = behavior_list.count(TacticalBehavior.LANE_CHANGE)
        total_car_following[parameter_set] = behavior_list.count(TacticalBehavior.CAR_FOLLOWING)

    sorted_collisions = sorted(total_collisions, key=total_collisions.get)
    sorted_collision_values = [total_collisions[k] for k in sorted_collisions]
    sorted_lc = sorted(total_lane_change, key=total_lane_change.get)
    sorted_lc_values=[total_lane_change[k] for k in sorted_lc]
    sorted_hl = sorted(total_lane_change, key=lambda k: total_lane_change.get(k) + total_car_following.get(k))
    sorted_hl_values = [total_lane_change[k] + total_car_following[k] for k in sorted_hl]

    print('most lane changes: ' + sorted_lc[-1])
    print('least collisions: ' + sorted_collisions[0])
    print('most human_like: ' + sorted_hl[-1])
