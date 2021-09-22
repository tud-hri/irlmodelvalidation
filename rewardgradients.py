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
from autograd import elementwise_grad, hessian


def calculate_new_state(x, u, dt):
    x_new = np.dot(np.array([[1, 0], [0, 1]]), x) + np.dot(np.array([[dt, 0], [0, dt]]), u)
    return x_new


def _gaussian_2d(x, y, sigma_x, sigma_y):
    x_part = (1 / (sigma_x * np.sqrt(2 * np.pi))) * np.exp(- (1 / 2) * (x ** 2 / sigma_x ** 2))
    y_part = (1 / (sigma_y * np.sqrt(2 * np.pi))) * np.exp(- (1 / 2) * (y ** 2 / sigma_y ** 2))

    return x_part * y_part


def calculate_g_and_H_autograd(constants, dt, N):
    def reward_function(u_h, theta, x_r, x0):
        x = [0.0] * 2 * (N + 1)
        x[0:2] = x0

        for n in range(1, N + 1):
            x[n * 2:(n + 1) * 2] = calculate_new_state(np.array(x[(n - 1) * 2:n * 2]), u_h[(n - 1) * 2: n * 2], dt)
        x = np.array(x)

        c2, v_desired, rb, lc, sigma_x, sigma_y, driving_direction = constants

        road_boundaries = 0.0
        lane_centers = 0.0
        velocity = 0.0
        collision = 0.0

        for n in range(1, N + 1):
            for boundary in rb:
                road_boundaries = road_boundaries + np.exp(-c2 * (x[1 + 2 * n] - boundary) ** 2)
            for center in lc:
                lane_centers = lane_centers + np.exp(-c2 * (x[1 + 2 * n] - center) ** 2)

            if driving_direction == 1:  # driving in top lanes, velocity is in negative x direction
                current_velocity = -1 * u_h[2 * (n - 1)]
            else:
                current_velocity = u_h[2 * (n - 1)]

            velocity = velocity + (current_velocity - v_desired) ** 2

            for x_opp in x_r:
                if all(x_opp[2 * n:2 + 2 * n]):
                    collision = collision + _gaussian_2d(x[2 * n] - x_opp[2 * n], x[1 + 2 * n] - x_opp[1 + 2 * n], sigma_x, sigma_y)

        return (theta * np.array([road_boundaries, lane_centers, velocity, collision])).sum()

    g = elementwise_grad(reward_function)
    H = hessian(reward_function)

    return g, H
