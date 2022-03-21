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

from dataobjects import HighDDataset
from tactical_evaluation import load_encrypted_pickle

data: HighDDataset


def get_examples_of_carfollowing(axis_item, dataset_id=12, colored_examples=3, grey_examples=52):
    data = load_encrypted_pickle('data/%02d.pkl' % dataset_id)
    count = 0
    target_number_of_examples = colored_examples + grey_examples

    for vehicle_id in data.track_meta_data.numFrames.sort_values(ascending=False).index:
        preceding_ids = data.track_data.loc[data.track_data['id'] == vehicle_id, 'precedingId'].unique()
        preceding_ids = preceding_ids[preceding_ids != 0]
        if len(preceding_ids) == 1 and data.track_meta_data.at[vehicle_id, 'numLaneChanges'] == 0:

            demo_data = data.track_data.loc[data.track_data['id'] == vehicle_id]
            ttc = demo_data['ttc'].to_numpy()
            inverse_ttc = 1 / ttc[ttc != 0]
            thw = demo_data['thw'].to_numpy()[ttc != 0]

            color = 'tab:green' if count < colored_examples else 'lightgrey'
            label ='Dataset %02d, Vehicle %d ' % (dataset_id, vehicle_id) if count < colored_examples else None
            line_width = 1. if count < colored_examples else 0.7
            z_order = 2 if count < colored_examples else 1

            axis_item.plot(thw, inverse_ttc, label=label, linewidth=line_width, color=color, zorder=z_order)

            if count < colored_examples:
                axis_item.plot(thw[-1], inverse_ttc[-1], color='tab:orange', marker='o', fillstyle='none', markersize=3., linestyle='', zorder=3)
                axis_item.plot(thw[0], inverse_ttc[0], color='k', marker='d', markersize=3., fillstyle='none', linestyle='', zorder=3)

            count += 1
            if count == target_number_of_examples:
                break
    axis_item.legend()

    return axis_item
