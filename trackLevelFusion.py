'''
This is a track level kalman filter, which is used fuse the tracked objects by lidar
with the information from another observation at asynchronized time.

The data fusion aglorithm will also need to accept a list of tracks (or None),
associating one or none of the tracks with the observation.

Model 1 Constant Velocity:
state vector: [x, y, vx, vy]
observation vector: [x, y, Vx, Vy ]

Model 2 Track Following:
state = [x,  y,  vx,  vy].T  
observation = 
        [distSF_mylap, velocity].T
'''
import numpy as np
import math
import pymap3d
import csv

from motionModel import ConstantVelocityFilter, MyLapsFilter
from enviroment import Point, PointEnvRaceTrack

class IMMFilter:
    '''
    The track level filter runs a IMM: Interactive Multi Model filter on top of two motiom models
    
    '''
    def __init__(self) -> None:
        enu_in = self.prepare_data('vegas_insideBounds.csv')
        enu_out = self.prepare_data('vegas_outsideBounds.csv')

        self.mylaps_filter = MyLapsFilter(enu_in, enu_out)
        self.lidar_filter = ConstantVelocityFilter(stateNoise=5,observationNoise=10)

    def update(self, lidar_tracks, myLaps_track, dt):
        '''
        Update using the filter with lidar data
        '''

        # Update mylaps filter anyway
        stateVector, stateCovariance, innovationCov = self.mylaps_filter.update(myLaps_track)

        # If there is a lidar track, pick a track, then update the lidar filter
        if lidar_tracks is not None:
            for i, track in enumerate(lidar_tracks) :
                # Choise the track which is closest to the myLaps track
                pass
            stateVector2 , stateCovariance2 , __ = self.lidar_filter.update(lidar_tracks)

    
    '''
    #Helper functions
    #Map origin:
        #   latitude: 36.27207268554108
        #   longitude: -115.0108130553903
        #   altitude: 594.9593907749116 (not used)
    '''
    def get_enu_from_csv(self, csv_file):
        with open(csv_file,newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = [row for row in reader]
        enu = []
        for latlong in data:
            latlong = [float(x) for x in latlong]
            enu.append( pymap3d.geodetic2enu(latlong[0],latlong[1],0,36.27207268554108,-115.0108130553903,0))
            
        enu = np.array(enu)[:,:2]
        return enu

    def get_distance_SF(self, enu):
        total_distance = 0
        distance_SF = [0]
        for i in range(len(enu)-1):
            distance = np.linalg.norm(enu[i+1]-enu[i])
            total_distance += distance
            distance_SF.append(total_distance)
        return np.array(distance_SF)[:,None]

    def prepare_data(self, csv_file):
        #read in the csv file
        enu_data = self.get_enu_from_csv(csv_file)
        #get the distance SF for each point
        distances_SF = self.get_distance_SF(enu_data)
        #combine the distance SF to enu points, NOTE this dosnt append the first point in the end
        return np.concatenate((enu_data,distances_SF),axis=1)