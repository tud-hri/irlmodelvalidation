"""
Copyright 2020, Olger Siebinga (o.siebinga@tudelft.nl)

This file is part of Travia.

Travia is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Travia is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Travia.  If not, see <https://www.gnu.org/licenses/>.
"""
import datetime
import sys
import os

from PyQt5 import QtWidgets

from dataobjects import HighDDataset
from dataobjects.enums import HighDDatasetID
from gui import TrafficVisualizerGui
from visualisation import HighDVisualisationMaster
from processing.encryptiontools import load_encrypted_pickle

if __name__ == '__main__':
    os.chdir(os.getcwd() + '/..')

    app = QtWidgets.QApplication(sys.argv)

    # Define the dataset used and the agent ID
    dataset_id = HighDDatasetID.DATASET_01
    dataset_index = dataset_id.value
    hga_agent_id = 118

    data = HighDDataset.load(dataset_id)

    if hga_agent_id is not None:
        simulation_dict = load_encrypted_pickle('data/HighD/data/%02d_agent_%d_simulated.pkl' % (dataset_index, hga_agent_id))
        first_frame = simulation_dict['first_frame']
        last_frame = simulation_dict['last_frame']
        agent_x = simulation_dict['agent_x']
        agent = simulation_dict['agent']

        print('%.2f; %.2f; %.2f; %.2f' % tuple(agent.theta))

        data.track_data = data.track_data.loc[(data.track_data['frame'] >= first_frame) & (data.track_data['frame'] <= last_frame), :]
        data.markers_data = []
        data.duration = (last_frame - first_frame) / data.frame_rate
        data.start_time += datetime.timedelta(microseconds=(first_frame / data.frame_rate) * 1e6)

        start_time = data.start_time
        end_time = start_time + datetime.timedelta(milliseconds=int(data.duration * 1000))
        number_of_frames = (last_frame - first_frame)
        dt = datetime.timedelta(seconds=1 / data.frame_rate)

        gui = TrafficVisualizerGui(data)
        sim = HighDVisualisationMaster(data, gui, start_time, end_time, number_of_frames, first_frame, dt, simulated_agent_data=agent_x, simulated_agent_original_id=hga_agent_id, simulated_agent=agent)
        gui.register_visualisation_master(sim)
    else:
        start_time = data.start_time
        end_time = start_time + datetime.timedelta(milliseconds=int(data.duration * 1000))
        first_frame = data.track_data['frame'].min()
        number_of_frames = data.track_data['frame'].max() - first_frame
        dt = datetime.timedelta(seconds=1 / data.frame_rate)

        gui = TrafficVisualizerGui(data)
        sim = HighDVisualisationMaster(data, gui, start_time, end_time, number_of_frames, first_frame, dt)
        gui.register_visualisation_master(sim)

    exit_code = app.exec_()
    sys.exit(exit_code)
