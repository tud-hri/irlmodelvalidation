import numpy as np
import time
import os
import glob
import multiprocessing as mp
from processing.encryptiontools import load_encrypted_pickle, save_encrypted_pickle


def compute_ttc(distance_gap, ego_v, lead_v):
    """
    Computes the time to collision

    :param distance_gap: distance gap over time
    :param ego_v: velocity of the ego vehicle
    :param lead_v: velocity of the lead vehicle
    :return:
    """
    return distance_gap / (ego_v - lead_v)


def compute_distance_gap(dhw: np.ndarray, length_ego_vehicle: float, length_opponent_vehicle: np.ndarray, driving_direction: float):
    """
    computes the distance gap (distance between front bumper of the ego vehicle and rear bumper of the lead vehicle)

    :param dhw: distance headway between rear bumpers, 0.0 indicates no valid leader
    :param length_ego_vehicle: in m
    :param length_opponent_vehicle: array with the length of the opponent vehicle on that frame, 0.0 if no opponent is present
    :param driving_direction: 1 for top lanes, 2 for bottom lanes
    :return:
    """
    if driving_direction == 2:
        return np.maximum(dhw - length_ego_vehicle, 0.0)
    else:
        return np.maximum(dhw - length_opponent_vehicle, 0.0)


def compute_time_gap(distance_gap, ego_velocity):
    """
    Computes the time headway (time between the rear and front bumpers of the vehicles).

    :param distance_gap: distance_gap over time in m
    :param ego_velocity: velocity of the ego vehicle over time in m/s
    :return:
    """
    return distance_gap/ego_velocity


def compute_dhw(ego_x, lead_x):
    """
    Computes the distance headway (distance between the rear or front bumpers of both vehicles, depends on the driving direction).

    :param ego_x: x position of the rear left corner of the ego vehicle over time
    :param lead_x: x position of the rear left corner of the lead vehicle over time, 0.0 indicates no valid leader
    :return:
    """
    return (lead_x != 0) * np.maximum(abs(lead_x - ego_x), 0.0)


def compute_thw(dhw, ego_velocity):
    """
    Computes the time headway (time between the rear or front bumpers of both vehicles, depends on the driving direction).

    :param dhw: dhw over time in m
    :param ego_velocity: velocity of the ego vehicle over time in m/s
    :return:
    """
    return dhw/ego_velocity


def compute_center_dhw(dhw, length_ego_vehicle, length_lead_vehicle, driving_direction):
    """
    Compute distance headway between centers of the ego and lead vehicle

    :param dhw: distance headway between rear bumpers, 0.0 indicates no valid leader
    :param length_ego_vehicle: in m
    :param length_lead_vehicle: in m
    :param driving_direction: 1 for top lanes, 2 for bottom lanes
    :return:
    """
    if driving_direction == 2:
        return np.maximum(dhw - (length_ego_vehicle / 2) + (length_lead_vehicle / 2), 0.0)
    else:
        return np.maximum(dhw + (length_ego_vehicle / 2) - (length_lead_vehicle / 2), 0.0)


def compute_and_save_metrics(dataset_index, filename, data):
    simulation_dict = load_encrypted_pickle(filename)

    agent_id = simulation_dict['agent_id']
    first_frame = simulation_dict['first_frame']
    last_frame = simulation_dict['last_frame']
    agent_x = simulation_dict['agent_x']
    agent = simulation_dict['agent']

    driving_direction = data.track_meta_data.at[agent_id, 'drivingDirection']

    all_frames = [n for n in range(first_frame, last_frame + 1)]

    ego_center_y = agent_x[:, 1] + agent.width / 2
    ego_lane_id = sum([ego_center_y > lm for lm in data.upper_lane_markings]) + sum([ego_center_y > lm for lm in data.lower_lane_markings]) + 1

    if driving_direction == 2:  # bottom lanes
        other_car_data = data.track_data.loc[
            (data.track_data['frame'].isin(all_frames)) & (data.track_data['id'] != agent_id) & (data.track_data['xVelocity'] > 0)]
    else:  # top lanes
        other_car_data = data.track_data.loc[
            (data.track_data['frame'].isin(all_frames)) & (data.track_data['id'] != agent_id) & (data.track_data['xVelocity'] < 0)]

    lead_id = np.array([0] * len(all_frames))
    lead_x = np.array([0.0] * len(all_frames))
    lead_v = np.array([0.0] * len(all_frames))
    length_lead_vehicle = np.array([0.0] * len(all_frames))

    for frame_index, frame in enumerate(all_frames):
        current_lane_id = ego_lane_id[frame_index]
        ego_x_position = agent_x[frame_index, 0]

        if driving_direction == 2:  # bottom lanes
            data_on_frame = other_car_data.loc[
                (other_car_data['frame'] == frame) & (other_car_data['laneId'] == current_lane_id) & (other_car_data['x'] > ego_x_position)]
        else:  # top lanes
            data_on_frame = other_car_data.loc[
                (other_car_data['frame'] == frame) & (other_car_data['laneId'] == current_lane_id) & (other_car_data['x'] < ego_x_position)]

        if len(data_on_frame):
            if driving_direction == 2:  # bottom lanes
                lead_id[frame_index] = data_on_frame.loc[data_on_frame['x'] == data_on_frame['x'].min(), 'id'].to_numpy()[0]
                lead_x[frame_index] = data_on_frame['x'].min()
            else:  # top lanes
                lead_id[frame_index] = data_on_frame.loc[data_on_frame['x'] == data_on_frame['x'].max(), 'id'].to_numpy()[0]
                lead_x[frame_index] = data_on_frame['x'].max()

            lead_v[frame_index] = data_on_frame.loc[data_on_frame['id'] == lead_id[frame_index], 'xVelocity'].to_numpy()[0]
            length_lead_vehicle[frame_index] = data.track_meta_data.loc[data.track_meta_data.index == lead_id[frame_index], 'width']

    if driving_direction == 1:
        agent_v = - agent_x[:, 2]
        lead_v *= -1
    else:
        agent_v = agent_x[:, 2]

    simulation_dict['dhw'] = compute_dhw(agent_x[:, 0], lead_x)
    simulation_dict['distance_gap'] = compute_distance_gap(simulation_dict['dhw'], agent.length, length_lead_vehicle, driving_direction)
    simulation_dict['center_dhw'] = compute_center_dhw(simulation_dict['dhw'], agent.length, length_lead_vehicle, driving_direction)
    simulation_dict['ttc'] = compute_ttc(simulation_dict['distance_gap'], agent_v, lead_v)
    simulation_dict['lane_id'] = ego_lane_id
    simulation_dict['delta_v'] = agent_v - lead_v
    simulation_dict['lead_id'] = lead_id
    simulation_dict['thw'] = compute_thw(simulation_dict['dhw'], agent_v)
    simulation_dict['time_gap'] = compute_time_gap(simulation_dict['distance_gap'], agent_v)

    save_encrypted_pickle(filename, simulation_dict)


def evaluate_one_dataset(all_agent_files, dataset_index):
    data = load_encrypted_pickle('data/HighD/data/%02d.pkl' % dataset_index)
    for file_name in all_agent_files:
        compute_and_save_metrics(dataset_index, file_name, data)

    print('Dataset %d is done' % dataset_index)


if __name__ == '__main__':
    os.chdir(os.getcwd() + '/..')
    t0 = time.time()
    arguments = []

    for data_id in range(1, 61):
        agent_files = glob.glob('data/HighD/data/%02d_agent_*_simulated.pkl' % data_id)
        if agent_files:
            arguments.append((agent_files, data_id))

    with mp.Pool(8) as p:
        p.starmap(evaluate_one_dataset, arguments)

    print('This took ' + str(time.time() - t0) + ' seconds')



