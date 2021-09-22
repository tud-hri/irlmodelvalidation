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
import autograd.numpy as np
from scipy import optimize


class IRLAgent:
    """
    This class represents human behavior in a driving game as was used in Sadigh et al. (2018).

    Differences from the paper implementation:
        - It neglects the term on heading since the high-D data set does not include headings
    """

    def __init__(self, width, length, theta, road_boundaries, lane_centers, desired_velocity, driving_direction, dt, N, c=0.18, sigma_x=10., sigma_y=1.4):
        self.width = width
        self.length = length
        self.theta = theta
        self.driving_direction = driving_direction
        self.road_boundaries = road_boundaries
        self.lane_centers = lane_centers

        self.last_u = np.array([0.0] * 2 * N)

        # Constants
        self.C2 = c
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.DESIRED_VELOCITY = desired_velocity

        self.N = N
        self.DT = dt

        self.min_x_acceleration = -6.63
        self.max_x_acceleration = 20.06
        self.max_y_acceleration = 1.63

    def calculate_action(self, x_0, surrounding_cars_positions, surrounding_car_sizes):
        result = optimize.minimize(self.calculate_summed_reward,
                                   np.array([0.0] * 2 * self.N),
                                   bounds=[(self.min_x_acceleration, self.max_x_acceleration), (-self.max_y_acceleration, self.max_y_acceleration)] * self.N,
                                   args=(x_0, surrounding_cars_positions, surrounding_car_sizes))
        self.last_u = result.x
        return self.last_u[0:2]

    def calculate_summed_reward(self, u, x_0, surrounding_cars_positions, surrounding_car_sizes, horizon=None):
        horizon = horizon if horizon else self.N

        summed_reward = 0.0
        x = x_0

        for n in range(horizon):
            distance_to_lane_center = abs(self.lane_centers - x[1])
            distance_to_road_boundary = abs(self.road_boundaries - x[1])

            if self.driving_direction == 1:  # driving in top lane with negative x velocity
                current_velocity = -1 * x[2]
            else:
                current_velocity = x[2]

            other_cars = surrounding_cars_positions[:, n, :]
            distances_to_other_cars = other_cars - x[0:2]

            summed_reward += self._calculate_reward(current_velocity, distance_to_lane_center, distance_to_road_boundary, distances_to_other_cars,
                                                    surrounding_car_sizes)
            x = self.calculate_new_state(x, np.array([u[2 * n], u[2 * n + 1]]), self.DT)
        return -summed_reward

    @staticmethod
    def calculate_new_state(x, u, dt):
        x_new = np.dot(np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]), x) + \
                np.dot(np.array([[0, 0], [0, 0], [dt, 0], [0, dt]]), u)
        return x_new

    def _calculate_reward(self, current_velocity, distance_to_lane_center, distance_to_road_boundary, distances_to_other_cars, surrounding_car_sizes):
        road_boundaries = sum(np.exp(-self.C2 * (distance_to_road_boundary ** 2)))
        lane_center = sum(np.exp(-self.C2 * (distance_to_lane_center ** 2)))
        velocity = (current_velocity - self.DESIRED_VELOCITY) ** 2
        collision = 0.0

        for index, distance_to_other_car in enumerate(distances_to_other_cars):
            collision += self._gaussian_2d(distance_to_other_car[0], distance_to_other_car[1], self.sigma_x, self.sigma_y)

        total_reward = self.theta[0] * road_boundaries + self.theta[1] * lane_center + self.theta[2] * velocity + self.theta[3] * collision
        return total_reward

    @staticmethod
    def _gaussian_2d(x, y, sigma_x, sigma_y):
        x_part = (1 / (sigma_x * np.sqrt(2 * np.pi))) * np.exp(- (1 / 2) * (x ** 2 / sigma_x ** 2))
        y_part = (1 / (sigma_y * np.sqrt(2 * np.pi))) * np.exp(- (1 / 2) * (y ** 2 / sigma_y ** 2))

        return x_part * y_part
