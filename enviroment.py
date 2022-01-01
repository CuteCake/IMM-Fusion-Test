'''
An env that generates a point moving on a curved line

This enviroment can deal with non stable update frequency, look at self.clock.tick(60)

It also outputs the observation

It also outputs the visualization of the enviroment
The visualization is a pygame window
Red dots are the observation, white dots are the groud truth



Author: Zhihao
Dependency: pygame,  installation see:https://www.pygame.org/wiki/GettingStarted
'''

from hashlib import new
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
    def __init__(self,enu_mat,distSF,velocity = 0):
        self.enu_mat = enu_mat
        total = np.linalg.norm(enu_mat[-1,:1]-enu_mat[0,:1])+enu_in[-1,2]
        self.enu_mat.append(np.array([[enu_mat[0,0],enu_mat[0,1],total]])) #add the first point to the end of the list
        print(self.enu_mat.shape)
        self.total_SF = total
        self.distSF = distSF
        self.velocity = velocity
        self.posx = 0
        self.posy = 0
        self.vx = 0
        self.vy = 0



    def update(self,dt): #This is a constant velocity, constant turning rate model, this update is for ground truth generation
        self.distSF += self.velocity*dt

        if self.distSF > self.total_SF:
            self.distSF -= self.total_SF

        for i, enu in enumerate(self.enu_mat):
            if enu[2] < self.distSF:
                self.posx = self._map(self.distSF, enu[2], self.enu_mat[i+1][2], enu[0], self.enu_mat[i+1][0])
                self.posy = self._map(self.distSF, enu[2], self.enu_mat[i+1][2], enu[1], self.enu_mat[i+1][1])

        self.velocity += random.gauss(0,2)
        

    def get_pos_from_distSF(self):
        for enu in self.enu_mat:
            if enu[2]>self.posx:
                break
        return (self.posx,self.posy)

    def _map(x, in_min, in_max, out_min, out_max):
        return ((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

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

def get_distance_SF(enu):
    total_distance = 0
    distance_SF = [0]
    for i in range(len(enu)-1):
        distance = np.linalg.norm(enu[i+1]-enu[i])
        total_distance += distance
        distance_SF.append(total_distance)
    return np.array(distance_SF)[:,None]

if __name__ == '__main__':

    #   latitude: 36.27207268554108
    #   longitude: -115.0108130553903
    #   altitude: 594.9593907749116
    enu_in=get_enu_from_csv('vegas_insideBounds.csv')
    enu_out=get_enu_from_csv('vegas_outsideBounds.csv')
    #get the distance SF for each point
    distance_SF_in = get_distance_SF(enu_in)
    distance_SF_out = get_distance_SF(enu_out)
    print(enu_in.shape,distance_SF_in.shape)
    #combine the distance SF to enu points
    enu_in = np.concatenate((enu_in,distance_SF_in),axis=1)
    enu_out = np.concatenate((enu_out,distance_SF_out),axis=1) #(n,3), axis 1 are: E, N, S/F distance

    # total_in = np.linalg.norm(enu_in[-1,:1]-enu_in[0,:1]) + enu_in[-1,2]
    # total_out = np.linalg.norm(enu_out[-1,:1]-enu_out[0,:1]) + enu_out[-1,2]

    # print(total_in,total_out)


    print(enu_in.shape,distance_SF_in.shape)

    enu_in_viz = enu_in[:,:2] + np.array([1920/2,1080/2])
    enu_out_viz = enu_out[:,:2] + np.array([1920/2,1080/2])
    

    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    # env = PointsEnv(640, 480, 10)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # print('time elapsed: ', env.get_time_elapsed())
                pygame.quit()
                quit()
        screen.fill((0, 0, 0))
        # env.draw(screen)
        # observation = env.update()
        # env.draw_observed_points(screen, observation)
        # print(env.getObservation())
        pygame.draw.aalines(screen, (255, 255, 255), True, enu_in_viz, 1)
        pygame.draw.circle(screen, (255, 255, 255), (int(enu_in_viz[0][0]),int(enu_in_viz[0][1])), 5)
        pygame.draw.aalines(screen, (255, 255, 255), True, enu_out_viz, 1)
        pygame.draw.circle(screen, (255, 255, 255), (int(enu_out_viz[0][0]),int(enu_out_viz[0][1])), 5)
        pygame.display.update()