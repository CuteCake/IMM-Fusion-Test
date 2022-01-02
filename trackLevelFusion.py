'''
This is a track level kalman filter, which is used fuse the tracked objects by lidar
with the information from another observation at asynchronized time.

The data fusion aglorithm will also need to accept a list of tracks (or None),
associating one or none of the tracks with the observation.

Model 1 Constant Velocity:
state vector: [x, y, vx, vy]
observation vector: [x, y, Vx, Vy ]

Model 2 Track Following:
state = [distSF_in, lateral, velocity, adjust].T
observation = 
        [distSF_mylap, velocity, x , y, v].T
'''
import numpy as np
import math

from motionModel import ConstantVelocityFilter, TrackFollowingFilter


class TrackLevelFilter:
    def __init__(self) -> None:
        pass