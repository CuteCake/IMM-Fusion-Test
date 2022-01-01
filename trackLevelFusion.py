'''
This is a track level kalman filter, which is used fuse the tracked objects by lidar
with the information from another observation at asynchronized time.

The data fusion aglorithm will also need to accept a list of tracks (or None),
associating one or none of the tracks with the observation.

state vector: [x, y, vx, vy]

observation vector: [x, y, Vx, Vy, ]
'''
import numpy as np
import math

from tracker import Track, MultiTracker
from motionModel import ConstantVelocityFilter

#This is the lookup table inputing the distance, output a position in ENU
LapDist2ENU_LUT = np.ones((360, 3))

class TrackLevelFilter:
    def __init__(self, ):
        self.observationMatrix = observationMatrix
        self.observationCovariance = observationCovariance
        self.stateTransitionCovariance = stateTransitionCovariance
        self.stateCovariance = stateCovariance
        self.stateNoise = stateNoise
        self.observationNoise = observationNoise
        self.id = id
        self.stateDim = len(self.stateCovariance)
        self.stateVector = np.zeros(self.stateDim)
        self.stateCovariance = np.eye(self.stateDim) * stateNoise*2   #TODO:  The initial state covariance is 2 times
                                             # the transition noise, for a better fix  in the future this should be adjustable
        self.stateUpdateMatrix = np.array([ [1, 0, 0, 0], \
                                            [0, 1, 0, 0], \
                                            [0, 0, 1, 0], \
                                            [0, 0, 0, 1]])
        

    def getStateUpdateMatrix(self, dt): #Get state estimation but don't update
        self.stateUpdateMatrix = np.array([ [1, 0, dt, 0], \
                                            [0, 1, 0, dt], \
                                            [0, 0, 1, 0], \
                                            [0, 0, 0, 1]])
        return self.stateUpdateMatrix

    def update_by_tracks(self, observation,  dt, observationCovariance=None ):
        observationMatrix = np.array([[1, 0, 0, 0], \

    def update_by_myLap(self, observation,  dt, observationCovariance=None ):
        observationMatrix = np.array([[1, 0, 0, 0], \
                                     [0, 1, 0, 0]])

    def getPrediction(self,  dt ):
        '''
        getPrediction step but don't change the state vector and the covariance matrix!
        This is useful for the gating step in association, or the publisher
        '''
        #Prediction step
        stateUpdateMatrix = self.getStateUpdateMatrix(dt)
        stateE = stateUpdateMatrix.dot(self.stateVector)
        stateCovarianceE = stateUpdateMatrix.dot(self.stateCovariance).dot(stateUpdateMatrix.T) + \
            self.stateTransitionCovariance

        obsE = self.observationMatrix.dot(stateE)

        return stateE, stateCovarianceE, obsE