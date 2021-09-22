import tqdm

from processing.encryptiontools import load_encrypted_pickle
from dataobjects import HighDDataset

if __name__ == '__main__':
    data: HighDDataset

    max_y_acc = 0.0
    max_x_acc = 0.0
    min_x_acc = 0.0

    total_vehicles = 0
    total_time = 0.
    road_length = []
    total_mean_v = 0.

    for index in tqdm.trange(1, 61):
        data = load_encrypted_pickle('../data/HighD/data/%02d.pkl' % index)

        max_y_acc_in_data = max(data.track_data['yAcceleration'].max(), -data.track_data['yAcceleration'].min())
        max_x_acc_in_data = data.track_data['xAcceleration'].max()
        min_x_acc_in_data = data.track_data['xAcceleration'].min()

        total_time += data.total_driven_time
        total_vehicles += data.num_vehicles
        road_length.append(float(data.track_data['x'].max()))

        total_mean_v += data.track_meta_data['meanXVelocity'].sum()

        if max_y_acc_in_data > max_y_acc:
            max_y_acc = max_y_acc_in_data
        if max_x_acc_in_data > max_x_acc:
            max_x_acc = max_x_acc_in_data
        if min_x_acc_in_data < min_x_acc:
            min_x_acc = min_x_acc_in_data

    print('max y = ', max_y_acc)
    print('max x = ', max_x_acc)
    print('min x = ', min_x_acc)

    print('average duration of a track = %.2f' % (total_time/total_vehicles))
    print('mean vehicle velocity = %.2f' % (total_mean_v / total_vehicles))
    print('average road length per recording = %.1f' % (sum(road_length) / len(road_length)))
