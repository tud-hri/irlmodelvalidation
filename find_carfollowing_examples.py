from dataobjects import HighDDataset
from tactical_evaluation import load_encrypted_pickle

data: HighDDataset


def get_examples_of_carfollowing(axis_item, dataset_id=12, target_number_of_examples=3):
    data = load_encrypted_pickle('data/%02d.pkl' % dataset_id)
    count = 0
    colors = ['tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red']

    for vehicle_id in data.track_meta_data.numFrames.sort_values(ascending=False).index:
        preceding_ids = data.track_data.loc[data.track_data['id'] == vehicle_id, 'precedingId'].unique()
        preceding_ids = preceding_ids[preceding_ids != 0]
        if len(preceding_ids) == 1 and data.track_meta_data.at[vehicle_id, 'numLaneChanges'] == 0:

            demo_data = data.track_data.loc[data.track_data['id'] == vehicle_id]
            ttc = demo_data['ttc'].to_numpy()
            inverse_ttc = 1 / ttc[ttc != 0]
            thw = demo_data['thw'].to_numpy()[ttc != 0]

            axis_item.plot(thw, inverse_ttc, label='Dataset %02d, Vehicle %d ' % (dataset_id, vehicle_id), linewidth=0.7, color=colors[count])
            axis_item.plot(thw[-1], inverse_ttc[-1], color='tab:orange', marker='o', fillstyle='none', markersize=3., linestyle='')
            axis_item.plot(thw[0], inverse_ttc[0], color='k', marker='d', markersize=3., fillstyle='none', linestyle='')
            count += 1
            if count == target_number_of_examples:
                break
    axis_item.legend()

    return axis_item
