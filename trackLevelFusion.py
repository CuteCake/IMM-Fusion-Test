'''
This is a track level kalman filter, which is used fuse the tracked objects by lidar
with the information from another observation at asynchronized time.

The data fusion aglorithm will also need to accept a list of tracks (or None),
associating one or none of the tracks with the observation.

Model 1 Constant Velocity:
state vector: [x, y, vx, vy].T
observation vector: [x, y, Vx, Vy ].T

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
        stateVector, stateCovariance, innovationCov = self.mylaps_filter.update(myLaps_track, dt)

        # If there is a lidar track, pick a track, then update the lidar filter
        if lidar_tracks is not None:

            # for i, track in enumerate(lidar_tracks) :
            #     # Choise the track which is closest to the myLaps track
            #     pass

            # Or, just pick the first track, under the assumption that the lidar tracks have no false positives, 
            # and the 
            stateVector2 , stateCovariance2 , __ = self.lidar_filter.update(lidar_tracks, dt)
            return stateVector2, stateCovariance2, innovationCov
        else:
            return stateVector, stateCovariance, innovationCov

    '''
    #Helper functions
    #Map origin:
        #   latitude: 36.27207268554108
        #   longitude: -115.0108130553903
        #   altitude: 594.9593907749116 (not used)

    ros__parameters: in GPS Fusion params: 
    dt: 0.01
    map_origin:
      latitude: 36.27207268554108
      longitude: -115.0108130553903
      altitude: 594.9593907749116
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


if __name__ == '__main__':
    from enviroment import *
    import time
    import matplotlib.pyplot as plt
    enu_in = prepare_data('vegas_insideBounds.csv')
    enu_out = prepare_data('vegas_outsideBounds.csv')

    #initialize visualization 
    enu_in_viz = enu_in[:,:2] + np.array([1920/2,1080/2])
    enu_out_viz = enu_out[:,:2] + np.array([1920/2,1080/2])
    
    
    env = PointEnvRaceTrack(1920, 1080, enu_in, enu_out, distSF_in=1000, lateral=10, velocity=50,adjust=1.03785)

    """"""
    #initialize the filter
    imm_filter = IMMFilter()

    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    start_time = time.time()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        #Step the environment
        screen.fill((0, 0, 0))
        obs_mylap, obs_lidar = env.step()
        if obs_lidar is not None:
            obs_lidar = obs_lidar[:2]
        env.draw(screen)

        if (time.time() - start_time)%30 > 5 :
            obs_mylap = None
            print('No mylap')
        # Use the kalman filter1
        stateVector, stateCovariance, innovationCov = imm_filter.update(obs_lidar, obs_mylap, env.get_last_dt())

        # Visualize the result 1
        xpos , ypos = stateVector[0], stateVector[1]
        xpos_viz = xpos + 1920/2
        ypos_viz = ypos + 1080/2
        pygame.draw.circle(screen, (0, 0, 255), (int(xpos_viz), int(ypos_viz)), 5)

        
        pygame.draw.aalines(screen, (255, 255, 255), True, enu_in_viz, 1)
        pygame.draw.circle(screen, (255, 255, 255), (int(enu_in_viz[0][0]),int(enu_in_viz[0][1])), 2)
        pygame.draw.aalines(screen, (255, 255, 255), True, enu_out_viz, 1)
        pygame.draw.circle(screen, (255, 255, 255), (int(enu_out_viz[0][0]),int(enu_out_viz[0][1])), 2)
        pygame.display.update()