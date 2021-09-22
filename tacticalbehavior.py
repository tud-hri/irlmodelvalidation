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
