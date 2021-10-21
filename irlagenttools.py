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
import numpy as np


def get_surrounding_cars_information(horizon_data, track_meta_data, first_frame, last_frame, agent_id):
    other_car_ids = horizon_data.loc[horizon_data['id'] == agent_id, 'precedingId'].unique().tolist() + \
                    horizon_data.loc[horizon_data['id'] == agent_id, 'followingId'].unique().tolist() + \
                    horizon_data.loc[horizon_data['id'] == agent_id, 'leftPrecedingId'].unique().tolist() + \
                    horizon_data.loc[horizon_data['id'] == agent_id, 'leftAlongsideId'].unique().tolist() + \
                    horizon_data.loc[horizon_data['id'] == agent_id, 'leftFollowingId'].unique().tolist() + \
                    horizon_data.loc[horizon_data['id'] == agent_id, 'rightPrecedingId'].unique().tolist() + \
                    horizon_data.loc[horizon_data['id'] == agent_id, 'rightAlongsideId'].unique().tolist() + \
                    horizon_data.loc[horizon_data['id'] == agent_id, 'rightFollowingId'].unique().tolist()

    other_car_ids = [i for i in other_car_ids if i != 0]

    surrounding_cars_positions = np.array([[[0.0, 0.0]] * (last_frame - first_frame + 1)] * len(other_car_ids))
    surrounding_car_sizes = np.array([[0.0, 0.0]] * len(other_car_ids))

    for index, other_car_id in enumerate(other_car_ids):
        other_car_length = track_meta_data.at[other_car_id, 'width']
        other_car_width = track_meta_data.at[other_car_id, 'height']

        surrounding_car_sizes[index, 0] = other_car_length
        surrounding_car_sizes[index, 1] = other_car_width

        other_car_data = horizon_data.loc[horizon_data['id'] == other_car_id, ['x', 'y']].to_numpy()
        other_car_first_frame = horizon_data.loc[horizon_data['id'] == other_car_id, 'frame'].min()
        other_car_last_frame = horizon_data.loc[horizon_data['id'] == other_car_id, 'frame'].max()

        other_car_data += np.array([[other_car_length / 2, other_car_width / 2]] * (other_car_last_frame - other_car_first_frame + 1))

        surrounding_cars_positions[index, other_car_first_frame - first_frame: other_car_last_frame - first_frame + 1, :] = other_car_data
    return surrounding_car_sizes, surrounding_cars_positions
