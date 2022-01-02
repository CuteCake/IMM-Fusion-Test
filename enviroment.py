'''
An env that generates a point moving on a curved line

This enviroment can deal with non stable update frequency, look at self.clock.tick(60)

It also outputs the observation

It also outputs the visualization of the enviroment
The visualization is a pygame window
Red dots are the observation, white dots are the groud truth



Author: Zhihao
Dependency: pygame,  installation see:https://www.pygame.org/wiki/GettingStarted
            pymap3d, use pip install pymap3d
'''

import random
import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

from pygame.event import get
import pymap3d
import csv

class Point:
    '''
    state = [distSF_in, lateral, velocity, adjust]          #(maybe add lag?)
        distSF: distance from the start/finish line of the track using INNER ring
                    the outer ring is 2*pi*15 = 94 m longer than the inner ring. in meters
                max: 2422.75 (length_in)
        lateral: the lateral distance from the inner ring of the track, in meters, 
                [min 0, max 15]
        velocity: the velocity of the car, m/s
                no limit
        adjust: the adjustment constant of distSF info from the my laps message, in respect to the inner ring
                    distSF_my_laps = distSF_inner_ring * adjust     
                [min 1, max 1.03785]

    state transition:
        distSF_in = distDF_in + velocity*dt
        lateral = lateral
        velocity = velocity
        adjust = adjust

    observation:
        obs_mylap = [distSF_mylap, velocity]
        obs_lidar_enu = [x,y,v] 
                #The tracked point in ENU frame, note the tracker outputs in baselink
                #Transform velocity from [vx,vy] to v

    observation function:
        distSF_mylap = distSF_in * adjust
        velocity_mylap = velocity         #up to 3% error, don't care
        x = from lookup table(state)
        y = from lookup table
        v = velocity

    enu_mat_in = :   the lookup table of the track
        [[x,y,distSF_in]
        [x,y,distSF_in]
        [x,y,distSF_in]
        ...
        [x,y,distSF_in]]
    '''
    def __init__(self,enu_mat_in, enu_mat_out, distSF_in = 0,lateral = 1,velocity = 0, adjust = 1.01):

        #Prepare the lookup table
        
        self.length_in = np.linalg.norm(enu_mat_in[-1,:1]-enu_mat_in[0,:1])+enu_mat_in[-1,2]#get the total length of the track
        last_enu = np.array([enu_mat_in[0,0],enu_mat_in[0,1],self.length_in])
        enu_mat_in = np.concatenate((enu_mat_in,last_enu[None,:]),axis=0) #add the first point to the end of the list


        last_enu = enu_mat_in[0,:]
        self.enu_mat_in = np.concatenate((enu_mat_in,last_enu[None,:]),axis=0)
        #Do it again for the outter ring

        self.length_out = np.linalg.norm(enu_mat_out[-1,:1]-enu_mat_out[0,:1])+enu_mat_out[-1,2]#get the total length of the track
        enu_mat_out =np.concatenate((enu_mat_out,np.array([enu_mat_out[0,0],enu_mat_out[0,1],self.length_out])[None,:]),axis=0) #add the first point to the end of the list
        last_enu = enu_mat_out[0,:]
        self.enu_mat_out = np.concatenate((enu_mat_out,last_enu[None,:]),axis=0)
        #Initialize the state
        self.distSF_in = distSF_in
        self.lateral = lateral
        self.velocity = velocity
        self.adjust = adjust


        #Initialize the observation from my lap
        self.distSF_mylap = self.distSF_in * self.adjust
        self.velocity = velocity

        #Initialize the output from Lidar
        self.posx = 0
        self.posy = 0
        self.vx = 0
        self.vy = 0

        #Other parameters might be needed
        self.track_width = 15   # 15 m track width
        self.in_out_ratio = 1.03785
        


    def update(self,dt): 
        '''
        Update the state of the enviroment and get the observation
        '''
        #Update the state
        self.distSF_in += self.velocity*dt
        if self.distSF_in > self.length_in:
            self.distSF_in -= self.length_in


        stateVec = [self.distSF_in,self.lateral,self.velocity,self.adjust]
        observation = self.observationFunc(stateVec)

        observation[2] += random.gauss(0,1)
        observation[3] += random.gauss(0,1)
        observation[4] += random.gauss(0,1)

        self.posx = observation[2]
        self.posy = observation[3]

        obs_mylap = observation[:2]
        obs_lidar_enu = observation[2:]

        # self.velocity += random.gauss(0,2)

        return obs_mylap, obs_lidar_enu
    
    def observationFunc(self,stateVector):
        '''
              state = [distSF_in, lateral, velocity, adjust].T
        observation = [distSF_mylap, velocity, x , y, v].T 
        Use the lookup table to get the observation
        '''
        
        distSF_in = stateVector[0]
        lateral = stateVector[1]
        velocity = stateVector[2]
        adjust = stateVector[3]

        #Get the my lap observation
        self.distSF_mylap = distSF_in * adjust
        self.velocity_mylap = velocity

        #Get the Lidar observation
            # enumerate through the enu look up table, find the entry with the closest distance, and interpolate the distance from 
            #point i and point i+1

        posx_in = 0
        posy_in = 0
        posx_out = 0
        posy_out = 0

        for i, enu in enumerate(self.enu_mat_in): #TODO Edge cases!
            # print(i,enu[2],self.distSF_in,self.length_in)
            # if enu[2] > self.distSF_in:
            if enu[2] >= distSF_in:
                posx_in = self._map(distSF_in, enu[2], self.enu_mat_in[i+1][2], enu[0], self.enu_mat_in[i+1][0])
                posy_in = self._map(distSF_in, enu[2], self.enu_mat_in[i+1][2], enu[1], self.enu_mat_in[i+1][1])
                break

        distance_SF_out = distSF_in * self.in_out_ratio
        if distance_SF_out > self.length_out:
            distance_SF_out -= self.length_out
        for i, enu in enumerate(self.enu_mat_out):
            if enu[2] >= distance_SF_out:
                posx_out = self._map(distance_SF_out, enu[2], self.enu_mat_out[i+1][2], enu[0], self.enu_mat_out[i+1][0])
                posy_out = self._map(distance_SF_out, enu[2], self.enu_mat_out[i+1][2], enu[1], self.enu_mat_out[i+1][1])
                break
        

        self.posx_lidar = self._map(lateral,0,self.track_width,posx_in,posx_out) 
        self.posy_lidar = self._map(lateral,0,self.track_width,posy_in,posy_out)
        self.v_lidar   = velocity

        observation = [self.distSF_mylap, velocity, self.posx_lidar,self.posy_lidar,self.v_lidar]
        return observation


    def _map(self, x, in_min, in_max, out_min, out_max):
        return ((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


class PointEnvRaceTrack:
    def __init__(self, width, height,enu_mat_in, enu_mat_out, distSF_in = 2410,lateral = 10,velocity = 100, adjust = 1.01):
        
        self.point = Point(enu_mat_in, enu_mat_out, distSF_in,lateral,velocity,adjust)

        self.width = width
        self.height = height

        self.clock = pygame.time.Clock()

        
    def step(self):
        dt = self.get_last_dt()
        self.obs_mylap, self.obs_lidar_enu = self.point.update(dt)

        self.clock.tick(60) #+random.randrange(-20,20)) #This limits The env to 60 frames per second by adding delay to the loop

        return self.obs_mylap, self.obs_lidar_enu

    def get_last_dt(self):
        return self.clock.get_time()/1000.0

    def draw(self, screen):

        point_X = self.point.posx + self.width/2
        point_Y = self.point.posy + self.height/2
        pygame.draw.circle(screen, (255,255,255), (int(point_X),int(point_Y)), 4)

        point_X_lidar = self.obs_lidar_enu[0] + self.width/2
        point_Y_lidar = self.obs_lidar_enu[1] + self.height/2
        pygame.draw.circle(screen, (255,0,0), (int(point_X_lidar),int(point_Y_lidar)), 4)

#Helper functions
#Map origin:
    #   latitude: 36.27207268554108
    #   longitude: -115.0108130553903
    #   altitude: 594.9593907749116 (not used)
def get_enu_from_csv(csv_file):
    with open(csv_file,newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]
    enu = []
    for latlong in data:
        latlong = [float(x) for x in latlong]
        enu.append( pymap3d.geodetic2enu(latlong[0],latlong[1],0,36.27207268554108,-115.0108130553903,0))
        
    enu = np.array(enu)[:,:2]
    return enu
    enu = enu + np.array([1920/2,1080/2])
    print(enu)
#helper function
def get_distance_SF(enu):
    total_distance = 0
    distance_SF = [0]
    for i in range(len(enu)-1):
        distance = np.linalg.norm(enu[i+1]-enu[i])
        total_distance += distance
        distance_SF.append(total_distance)
    return np.array(distance_SF)[:,None]

def prepare_data(csv_file):
    #read in the csv file
    enu_data = get_enu_from_csv(csv_file)
    #get the distance SF for each point
    distances_SF = get_distance_SF(enu_data)
    #combine the distance SF to enu points, NOTE this dosnt append the first point in the end
    return np.concatenate((enu_data,distances_SF),axis=1)

if __name__ == '__main__':

    enu_in = prepare_data('vegas_insideBounds.csv')
    enu_out = prepare_data('vegas_outsideBounds.csv')

    #initialize visualization 
    enu_in_viz = enu_in[:,:2] + np.array([1920/2,1080/2])
    enu_out_viz = enu_out[:,:2] + np.array([1920/2,1080/2])
    
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    env = PointEnvRaceTrack(1920, 1080, enu_in, enu_out)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        screen.fill((0, 0, 0))
        obs_mylap, obs_lidar = env.step()
        env.draw(screen)
        
        pygame.draw.aalines(screen, (255, 255, 255), True, enu_in_viz, 1)
        pygame.draw.circle(screen, (255, 255, 255), (int(enu_in_viz[0][0]),int(enu_in_viz[0][1])), 5)
        pygame.draw.aalines(screen, (255, 255, 255), True, enu_out_viz, 1)
        pygame.draw.circle(screen, (255, 255, 255), (int(enu_out_viz[0][0]),int(enu_out_viz[0][1])), 5)
        pygame.display.update()