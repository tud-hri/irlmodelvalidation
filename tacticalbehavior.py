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
import enum


class TacticalBehavior(enum.Enum):
    LANE_CHANGE = 0
    CAR_FOLLOWING = 1
    COLLISION = 2
    OFF_ROAD = 3

    def __str__(self):
        return {TacticalBehavior.LANE_CHANGE: 'Lane Change',
                TacticalBehavior.CAR_FOLLOWING: 'Car Following',
                TacticalBehavior.COLLISION: 'Collision',
                TacticalBehavior.OFF_ROAD: 'Off Road'
                }[self]
