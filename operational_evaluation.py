import glob
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import tqdm
from scipy import stats

from find_carfollowing_examples import get_examples_of_carfollowing
from processing.encryptiontools import load_encrypted_pickle
from tacticalbehavior import TacticalBehavior

if __name__ == '__main__':
    os.chdir(os.getcwd() + '/..')
    plt.rcParams.update({'font.size': 14})

    all_agent_files = []
    all_demo_end_points = []
    all_demo_start_points = []
    all_lc_agent_end_points = []
    all_lc_agent_start_points = []
    all_cf_agent_start_points = []

    ttc_difference = []
    time_difference = []
    thw_difference = []
    ttc_difference_percentage = []
    phase_plane_errors_end_point_distance = []
    filtered_cf = []
    cf = []
    perfect_times = []
    small_difference_lc = []

    for dataset_index in range(1, 61):
        all_agent_files += glob.glob('data/HighD/data/%02d_agent_*_simulated.pkl' % dataset_index)

    lane_change_window = plt.figure('All Lane Changes')
    demo_plot = lane_change_window.add_subplot(2, 2, 1, title='Human \n a.', ylabel='Lane Change \n $TTC^{-1} [s^{-1}]$')
    lane_change_plot = lane_change_window.add_subplot(2, 2, 2, title='Model \n b.')
    demo_car_following_plot = lane_change_window.add_subplot(2, 2, 3, title='c.', xlabel='Time gap [s]',
                                                             ylabel='Car following \n $TTC^{-1} [s^{-1}]$')
    car_following_plot = lane_change_window.add_subplot(2, 2, 4, title='d.', xlabel='Time gap [s]')

    diff_window = plt.figure('time diff')
    diff_plots = diff_window.add_subplot(1, 2, 1, title='Absolute differences between agents and their \n demonstrations at the moment of lane change',
                                         ylabel='ttc difference [s]', xlabel='time gap difference [s]')
    small_error_plot = diff_window.add_subplot(1, 2, 2, title='Agent and demonstration lane changes \n where the difference is < 0.5 s ',
                                               xlabel='Time gap [s]', ylabel='$TTC^{-1} [s^{-1}]$')

    last_dataset_id = 1
    data = load_encrypted_pickle('data/HighD/data/%02d.pkl' % last_dataset_id)

    for agent_file_name in tqdm.tqdm(all_agent_files):

        agent_data = load_encrypted_pickle(agent_file_name)

        dataset_id = agent_data['dataset_id']
        agent_id = agent_data['agent_id']
        category = agent_data['tactical_behavior']

        if last_dataset_id != dataset_id:
            last_dataset_id = dataset_id
            data = load_encrypted_pickle('data/HighD/data/%02d.pkl' % dataset_id)

        demonstration_data = data.track_data.loc[data.track_data['id'] == agent_id]

        # find when the demo performs its lane change
        demo_frame_index_of_lc = int(demonstration_data.loc[(demonstration_data['laneId'] - demonstration_data['laneId'].shift(-1)).abs() == 1, 'frame'])

        # initialize agent lane change and collision at the last frame, then try to find if it happened
        agent_frame_index_of_lc = len(agent_data['lane_id'])
        agent_frame_index_of_collision = len(agent_data['lane_id'])
        last_plot_frame_index = len(agent_data['lane_id'])

        demo_ttc = demonstration_data.loc[
            (demonstration_data['frame'] >= agent_data['first_frame']) & (demonstration_data['frame'] <= demo_frame_index_of_lc), 'ttc'].to_numpy()
        demo_leading_vehicle = demonstration_data.loc[
            (demonstration_data['frame'] >= agent_data['first_frame']) & (demonstration_data['frame'] <= demo_frame_index_of_lc), 'precedingId'].unique()
        if any(demo_ttc == 0):
            # the lane change happened when the preceding car is out of view, resulting in an unknown ttc at lc time, do not plot this vehicle
            continue
        elif len(demo_leading_vehicle) > 1:
            # the vehicle has multiple preceding vehicles, this results in jumps in the plot and thus it is filtered out
            continue
        demo_inverse_ttc = 1. / demo_ttc
        demo_thw = demonstration_data.loc[
            (demonstration_data['frame'] >= agent_data['first_frame']) & (demonstration_data['frame'] <= demo_frame_index_of_lc), 'thw'].to_numpy()

        if agent_data['tactical_behavior'] == TacticalBehavior.LANE_CHANGE:
            agent_frame_index_of_lc = np.where((agent_data['lane_id'] - np.roll(agent_data['lane_id'], -1)) != 0)[0][0] + 1
            last_plot_frame_index = agent_frame_index_of_lc

            agent_ttc = agent_data['ttc'][0: last_plot_frame_index]
            agent_leading_vehicle = np.unique(agent_data['lead_id'][0: last_plot_frame_index])
            if any(agent_ttc == 0):
                # the lane change happened when the preceding car is out of view, resulting in an unknown ttc at lc time, do not plot this vehicle
                continue
            elif len(agent_leading_vehicle) > 1:
                # the vehicle has multiple preceding vehicles, this results in jumps in the plot and thus it is filtered out
                continue
            agent_inverse_ttc = 1. / agent_ttc
            agent_time_gap = agent_data['time_gap'][0: last_plot_frame_index]

            plot_item = lane_change_plot
            plot_item.plot(agent_time_gap, agent_inverse_ttc, color='tab:blue', linewidth=0.7)

            phase_plane_errors_end_point_distance.append(np.sqrt((agent_time_gap[-1] - demo_thw[-1]) ** 2 + (agent_ttc[-1] - demo_ttc[-1]) ** 2))
            all_lc_agent_end_points.append([agent_time_gap[-1], agent_inverse_ttc[-1]])
            all_lc_agent_start_points.append([agent_time_gap[0], agent_inverse_ttc[0]])
            ttc_difference.append(demo_ttc[-1] - agent_ttc[-1])

            time_difference.append(demo_frame_index_of_lc - (agent_data['first_frame'] + agent_frame_index_of_lc))
            ttc_difference_percentage.append((demo_ttc[-1] - agent_ttc[-1]) * 100 / demo_ttc[-1])
            thw_difference.append(demo_thw[-1] - agent_time_gap[-1])
            demo_plot.plot(demo_thw, demo_inverse_ttc, color='tab:blue', linewidth=0.7)

            all_demo_end_points.append([demo_thw[-1], demo_inverse_ttc[-1]])
            all_demo_start_points.append([demo_thw[0], demo_inverse_ttc[0]])

            if np.linalg.norm(np.array([thw_difference[-1], ttc_difference[-1]])) < 0.5:
                small_error_last_agent_line = small_error_plot.plot(agent_time_gap, agent_inverse_ttc, color='tab:blue', linewidth=0.7)
                small_error_last_demo_line = small_error_plot.plot(demo_thw, demo_inverse_ttc, color='tab:orange', linewidth=0.7)

                small_error_last_end = small_error_plot.plot(agent_time_gap[-1], agent_inverse_ttc[-1], color='k', marker='d', fillstyle='none', markersize=3.,
                                                             linestyle='')
                small_error_plot.plot(demo_thw[-1], demo_inverse_ttc[-1], color='k', marker='d', fillstyle='none', markersize=3., linestyle='')
                small_error_last_start = small_error_plot.plot(agent_time_gap[0], agent_inverse_ttc[0], color='tab:orange', marker='o', fillstyle='none',
                                                               markersize=3.,
                                                               linestyle='')

        elif agent_data['tactical_behavior'] == TacticalBehavior.CAR_FOLLOWING:
            agent_ttc = agent_data['ttc']
            agent_leading_vehicle = np.unique(agent_data['lead_id'])

            if len(agent_leading_vehicle[agent_leading_vehicle != 0]) > 1:
                # the vehicle has multiple preceding vehicles, this results in jumps in the plot and thus it is filtered out
                filtered_cf.append(agent_file_name)
                continue
            else:
                cf.append(agent_file_name)

            agent_inverse_ttc = 1. / agent_ttc[agent_ttc != 0]
            agent_time_gap = agent_data['time_gap'][agent_ttc != 0]

            car_following_plot.plot(agent_time_gap, agent_inverse_ttc, color='tab:green', linewidth=0.7)
            all_cf_agent_start_points.append([agent_time_gap[0], agent_inverse_ttc[0]])

    all_lc_agent_end_points = np.array(all_lc_agent_end_points)
    all_lc_agent_start_points = np.array(all_lc_agent_start_points)
    all_cf_agent_start_points = np.array(all_cf_agent_start_points)
    all_demo_end_points = np.array(all_demo_end_points)
    all_demo_start_points = np.array(all_demo_start_points)

    # calculate number of absolute ttc/time gap differences smaller then 1 second
    total_okeisch_lc = 0
    for point in zip(abs(np.array(thw_difference)), abs(np.array(ttc_difference))):
        if np.linalg.norm(point) < .5:
            total_okeisch_lc += 1

    print('A total of %d lane changes by the agents had a difference with the human demo of less then 1 second in ttc/time gap.' % total_okeisch_lc)

    demo_plot.plot(all_demo_end_points[:, 0], all_demo_end_points[:, 1], color='tab:orange', marker='o', fillstyle='none', markersize=3., linestyle='')
    demo_plot.plot(all_demo_start_points[:, 0], all_demo_start_points[:, 1], color='k', marker='d', fillstyle='none', markersize=3., linestyle='')

    lane_change_plot.plot(all_lc_agent_end_points[:, 0], all_lc_agent_end_points[:, 1], color='tab:orange', marker='o', fillstyle='none', markersize=3.,
                          linestyle='')
    lane_change_plot.plot(all_lc_agent_start_points[:, 0], all_lc_agent_start_points[:, 1], color='k', fillstyle='none', marker='d', markersize=3.,
                          linestyle='')

    car_following_plot.plot(all_cf_agent_start_points[:, 0], all_cf_agent_start_points[:, 1], color='k', fillstyle='none', marker='d', markersize=3.,
                            linestyle='')

    demo_plot.hlines(0.0, -10, 10, linestyles='dashed', color='#c9c9c9')
    demo_plot.vlines(0.0, -10, 10, linestyles='dashed', color='#c9c9c9')
    lane_change_plot.hlines(0.0, -10, 10, linestyles='dashed', color='#c9c9c9')
    lane_change_plot.vlines(0.0, -10, 10, linestyles='dashed', color='#c9c9c9')
    car_following_plot.hlines(0.0, -10, 10, linestyles='dashed', color='#c9c9c9')
    car_following_plot.vlines(0.0, -10, 10, linestyles='dashed', color='#c9c9c9')
    demo_car_following_plot.hlines(0.0, -10, 10, linestyles='dashed', color='#c9c9c9')
    demo_car_following_plot.vlines(0.0, -10, 10, linestyles='dashed', color='#c9c9c9')

    # plot circle
    phi = np.linspace(0, 2 * np.pi, 1000)

    small_error_plot.hlines(0.0, -11, 11, linestyles='dashed', color='#c9c9c9')
    small_error_plot.vlines(0.0, -11, 11, linestyles='dashed', color='#c9c9c9')

    orange_line = mlines.Line2D([], [], color='tab:orange', linewidth=.7, label='Human demonstrations')
    blue_line = mlines.Line2D([], [], color='tab:blue', linewidth=.7, label='Agent behavior')
    end_marker = mlines.Line2D([], [], color='tab:orange', marker='o', fillstyle='none', markersize=3., linestyle='', label='Lane Change')
    start_marker = mlines.Line2D([], [], color='k', marker='d', fillstyle='none', markersize=3., linestyle='', label='Initial point')
    small_error_plot.legend(handles=[orange_line, blue_line, start_marker, end_marker])
    lane_change_plot.legend(handles=[start_marker, end_marker])
    demo_plot.legend(handles=[start_marker, end_marker])

    get_examples_of_carfollowing(demo_car_following_plot)

    demo_plot.set_xlim([-0.5, 5.])
    demo_plot.set_ylim([-0.1, 1.0])
    lane_change_plot.set_xlim([-0.5, 5.])
    lane_change_plot.set_ylim([-0.1, 1.0])
    car_following_plot.set_xlim([-0.5, 5.])
    car_following_plot.set_ylim([-0.3, 0.8])
    demo_car_following_plot.set_xlim([-0.5, 5.])
    demo_car_following_plot.set_ylim([-0.3, 0.8])

    diff_plots.set_xlim([-0.5, 3])
    diff_plots.set_ylim([-0.5, 10])

    small_error_plot.set_xlim([-0.1, 3.5])
    small_error_plot.set_ylim([-0.1, 0.65])

    demo_violin_ttc_data = pd.DataFrame(columns=['source', 'data_point', 'label'])
    demo_violin_ttc_data['data_point'] = all_demo_end_points[:, 1]
    demo_violin_ttc_data['source'] = 'Human'
    demo_violin_ttc_data['label'] = 'Inverse TTC'

    demo_violin_thw_data = pd.DataFrame(columns=['source', 'data_point', 'label'])
    demo_violin_thw_data['data_point'] = all_demo_end_points[:, 0]
    demo_violin_thw_data['source'] = 'Human'
    demo_violin_thw_data['label'] = 'Time Gap'

    agent_violin_ttc_data = pd.DataFrame(columns=['source', 'data_point', 'label'])
    agent_violin_ttc_data['data_point'] = all_lc_agent_end_points[:, 1]
    agent_violin_ttc_data['source'] = 'Model'
    agent_violin_ttc_data['label'] = 'Inverse TTC'

    agent_violin_thw_data = pd.DataFrame(columns=['source', 'data_point', 'label'])
    agent_violin_thw_data['data_point'] = all_lc_agent_end_points[:, 0]
    agent_violin_thw_data['source'] = 'Model'
    agent_violin_thw_data['label'] = 'Time Gap'

    data_for_violin_plot = pd.DataFrame(columns=['source', 'data_point', 'label'])
    data_for_violin_plot = data_for_violin_plot.append(demo_violin_ttc_data)
    data_for_violin_plot = data_for_violin_plot.append(demo_violin_thw_data)
    data_for_violin_plot = data_for_violin_plot.append(agent_violin_ttc_data)
    data_for_violin_plot = data_for_violin_plot.append(agent_violin_thw_data)

    plt.figure()
    seaborn.violinplot(data=data_for_violin_plot, y='label', x='data_point', hue='source', split=True, scale_hue=True, scale="area", inner="quartile", cut=0,
                       bw='silverman')
    plt.xlabel('Inverse TTC [1/s] or Time Gap [s]')
    plt.xlim([-0.25, 4.])
    plt.title('Estimated distributions of the inverse ttc and time gap at moment of lane change')

    demo_ttc_t_test = demo_violin_ttc_data['data_point'].to_numpy()
    agent_ttc_t_test = agent_violin_ttc_data['data_point'].to_numpy()
    demo_thw_t_test = demo_violin_thw_data['data_point'].to_numpy()
    agent_thw_t_test = agent_violin_thw_data['data_point'].to_numpy()

    thw_t_test_results = stats.ttest_rel(demo_thw_t_test, agent_thw_t_test)
    ttc_t_test_results = stats.ttest_rel(demo_ttc_t_test, agent_ttc_t_test)

    print('t-test for thw: statitic = %.3f, p-value = %.3e' % (thw_t_test_results.statistic, thw_t_test_results.pvalue))
    print('t-test for ttc: statitic = %.3f, p-value = %.3e' % (ttc_t_test_results.statistic, ttc_t_test_results.pvalue))

    plt.show()
