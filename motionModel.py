'''
This files stores a bunch of Kalman Filters for the tracker.

If we use the most basic constant velocity model, we don't need to care about reference frame
too much, but make sure the relative velocity is not too fast or it might have problem with
data association.

If we use the constant velocity constant turning rate model, we need to do the tracking in 
earth / map / ENU reference frame.

Author: Zhihao
'''

import numpy as np
import math

class BaseFilter: #This is a template for motion filters, should be overwritten
    def __init__(self):
        pass
        
    def update(self, observation, dt):
        raise NotImplementedError

class ConstantVelocityFilter(BaseFilter): 
    '''
    Constent velocity model Kalman Filter, not EKF!
    The state space is defined as:
    [x, y, vx, vy]: x,y are the position, vx,vy are the velocities

    The observation space is defined as:
    [x, y]

    The state transition model is:
    x(k+1) = x(k) + vx(k) * dt
    y(k+1) = y(k) + vy(k) * dt
    vx(k+1) = vx(k)
    vy(k+1) = vy(k)
    It is linear, that's why it can be represented as a matrix.
    But in CVCT it can only be represented as a bunch of equations.

    The observation model is:
    x(k) = x(k)
    y(k) = y(k)

    It is linear, that's why it can be represented as a matrix.
    '''
    def __init__(self, x=0, y=0, vx=0, vy=0, \
        stateNoise=0.5,observationNoise=10, stateInitNoise=1):
        #state variables in Numpy array
        #[x, y, vx, vy].T

        self.stateVector = np.array([x, y, vx, vy]).T #It is a column vector
        self.stateDim = len(self.stateVector) 
        self.stateTransitionCovariance = \
            np.array(   [[1, 0, 0, 0], \
                        [0, 1, 0, 0], \
                        [0, 0, 1, 0], \
                        [0, 0, 0, 1], ]) * stateNoise
        self.observationCovariance = \
            np.array(   [[1, 0], \
                        [0, 1], ]) * observationNoise
        self.observationMatrix = np.array([[1, 0, 0, 0], \
                                           [0, 1, 0, 0]])
        self.stateCovariance = np.eye(4) * stateNoise*2   #TODO:  The initial state covariance is 2 times
                                            # the transition noise, for a better fix  in the future this should be adjustable
        self.observationCovariance = np.eye(2) * observationNoise
        self.id = id

    def getStateUpdateMatrix(self, dt): #Get state estimation but don't update
        self.stateUpdateMatrix = np.array([ [1, 0, dt, 0], \
                                            [0, 1, 0, dt], \
                                            [0, 0, 1, 0], \
                                            [0, 0, 0, 1]])
        return self.stateUpdateMatrix


    def update(self, observation,  dt, observationCovariance=None ):
        if observationCovariance is not None:
            self.observationCovariance = observationCovariance
        '''
        observation: [x,y]
        dt: time since last update
        '''
        #Prediction step
        stateUpdateMatrix = self.getStateUpdateMatrix(dt)
        stateE = stateUpdateMatrix.dot(self.stateVector)
        stateCovarianceE = stateUpdateMatrix.dot(self.stateCovariance).dot(stateUpdateMatrix.T) + \
            self.stateTransitionCovariance

        if observation is None:
            self.state = stateE
            self.stateCovariance = stateCovarianceE
            return self.state, self.stateCovariance, np.zeros_like(self.observationCovariance)
        #Generate Kalman Gain

        innovationCov = self.observationCovariance + \
            self.observationMatrix.dot(stateCovarianceE).dot(self.observationMatrix.T)

        kalmanGain = stateCovarianceE.dot(self.observationMatrix.T).dot(np.linalg.inv(innovationCov))

        # kalmanGain = stateCovarianceE.dot(self.observationMatrix.T).dot(np.linalg.inv(self.observationCovariance + \
        #     self.observationMatrix.dot(stateCovarianceE).dot(self.observationMatrix.T)))
        #Correct prediction
        self.stateVector = stateE + kalmanGain.dot(np.array(observation).T - self.observationMatrix.dot(stateE))
        self.stateCovariance = (np.eye(self.stateDim) - kalmanGain.dot(self.observationMatrix)).dot(stateCovarianceE)
        return self.stateVector, self.stateCovariance, innovationCov

    def prediction(self, dt):
        '''
        prediction step
        Why split up? for multi object tracking, because sometimes
        we don't get an observation, we will not call the correction(), 
        but we still need to predict

        This step changes the state vector and the covariance matrix!

        so calling prediction() and correction() in a row 
        will give the same result as calling update()
        '''
        #Prediction step
        stateUpdateMatrix = self.getStateUpdateMatrix(dt)
        stateE = stateUpdateMatrix.dot(self.stateVector)
        stateCovarianceE = stateUpdateMatrix.dot(self.stateCovariance).dot(stateUpdateMatrix.T) + \
            self.stateTransitionCovariance

        self.stateVector = stateE
        self.stateCovariance = stateCovarianceE

        predictedObservation = self.observationMatrix.dot(self.stateVector)

        return self.stateVector, self.stateCovariance, predictedObservation

    def correction(self, observation, observationCovariance=None):
        '''
        correction step
        Why split up? for multi object tracking!
        '''
        if observationCovariance is not None:
            self.observationCovariance = observationCovariance
        #get back the estimation
        stateCovarianceE = self.stateCovariance
        stateE= self.stateVector
        #Generate Kalman Gain
        kalmanGain = stateCovarianceE.dot(self.observationMatrix.T).dot(np.linalg.inv(self.observationCovariance + \
            self.observationMatrix.dot(stateCovarianceE).dot(self.observationMatrix.T)))
        #Correct prediction
        self.stateVector = stateE + kalmanGain.dot(np.array(observation).T - self.observationMatrix.dot(stateE))
        self.stateCovariance = (np.eye(self.stateDim) - kalmanGain.dot(self.observationMatrix)).dot(stateCovarianceE)
        return self.stateVector

    def getPrediction(self, dt):
        '''
        getPrediction step but don't change the state vector and the covariance matrix!
        '''
        #Prediction step
        stateUpdateMatrix = self.getStateUpdateMatrix(dt)
        stateE = stateUpdateMatrix.dot(self.stateVector)
        stateCovarianceE = stateUpdateMatrix.dot(self.stateCovariance).dot(stateUpdateMatrix.T) + \
            self.stateTransitionCovariance

        obsE = self.observationMatrix.dot(stateE)

        return stateE, stateCovarianceE, obsE
    
class MyLapsFilter(BaseFilter):
    '''
    This is a EKF only for My Laps system

    state = [x,  y,  vx,  vy].T          
 

    state transition:
        [1, 0, dt, 0], \
        [0, 1, 0, dt], \
        [0, 0, 1, 0], \
        [0, 0, 0, 1]

    #Consider using look up table for state transition? Not do it for now cuz it might cause convergance problem

    observation = 
        [x_m, y_m, v_m].T  

        obs_mylap = [distSF_mylap, velocity], -> [x_m, y_m, v_m] (by look up table)
        obs_lidar_enu = [x,y,vx,vy] 
                #The tracked point in ENU frame, note the tracker outputs in baselink


    observation transformation by look up table:
        [distSF_mylap, velocity] -> [x_mylap, y_mylap, vx_mylap, vy,_mylap]
        There are some parameters to get:
            distSF: distance from the start/finish line of the track using INNER ring
                        the outer ring is 2*pi*15 = 94 m longer than the inner ring. in meters
                    max: 2422.75 (length_in)
            lateral: the lateral distance from the inner ring of the track, in meters, 
        lateral: the lateral distance from the inner ring of the track, in meters, 
            lateral: the lateral distance from the inner ring of the track, in meters, 
        lateral: the lateral distance from the inner ring of the track, in meters, 
            lateral: the lateral distance from the inner ring of the track, in meters, 
        lateral: the lateral distance from the inner ring of the track, in meters, 
            lateral: the lateral distance from the inner ring of the track, in meters, 
                    [min 0, max 15]
            velocity: the velocity of the car, m/s
                    no limit
            adjust: the adjustment constant of distSF info from the my laps message, in respect to the inner ring
                        distSF_my_laps = distSF_inner_ring * adjust     
                    distSF_my_laps = distSF_inner_ring * adjust     
                        distSF_my_laps = distSF_inner_ring * adjust     
                    distSF_my_laps = distSF_inner_ring * adjust     
                        distSF_my_laps = distSF_inner_ring * adjust     
                    distSF_my_laps = distSF_inner_ring * adjust     
                        distSF_my_laps = distSF_inner_ring * adjust     
                    [min 1, max 1.03785]

    observation function:
        x_m = x
        y_m = y
        v_m = sqrt(vx^2 + vy^2)

    observation jacobian matrix:
        [1, 0, 0, 0], \
        [0, 1, 0, 0], \
        [0, 0, vx/sqrt(vx^2+vy^2), vx/sqrt(vx^2+vy^2)]

    enu_mat_in = :   the lookup table of the track
        [[x,y,distSF_in]
        [x,y,distSF_in]
        [x,y,distSF_in]
        ...
        [x,y,distSF_in]]
    '''

    def __init__(self, enu_mat_in, enu_mat_out, init_state = [0,0,0,0], stateNoise=1, observationNoise=1):
        #Prepare the lookup table
        
        self.length_in = np.linalg.norm(enu_mat_in[-1,:2]-enu_mat_in[0,:2])+enu_mat_in[-1,2]#get the total length of the track
        enu_mat_in =np.concatenate((enu_mat_in,np.array([enu_mat_in[0,0],enu_mat_in[0,1],self.length_in])[None,:]),axis=0)
        last_enu = enu_mat_in[0,:]
        self.enu_mat_in = np.concatenate((enu_mat_in,last_enu[None,:]),axis=0)
        self.enu_mat_in = enu_mat_in
        #Do it again for the outter ring
        self.length_out = np.linalg.norm(enu_mat_out[-1,:2]-enu_mat_out[0,:2])+enu_mat_out[-1,2]#get the total length of the track
        enu_mat_out =np.concatenate((enu_mat_out,np.array([enu_mat_out[0,0],enu_mat_out[0,1],self.length_out])[None,:]),axis=0) #add the first point to the end of the list
        last_enu = enu_mat_out[0,:]
        self.enu_mat_out = np.concatenate((enu_mat_out,last_enu[None,:]),axis=0)

        #Initialize the state
        self.state = np.array(init_state).T


        #Initialize the transformed observation from MyLaps system
        self.x_m = 0
        self.y_m = 0
        self.v_m = 0

        #Other parameters might be needed
        self.track_width = 15   # 15 m track width
        self.in_out_ratio = self.length_out/self.length_in
        self.stateDim = len(self.state)


        #Noise assumption about state transition and observation

        # state =                   [x,  y,  vx,  vy].T  
        self.state_tran_noise_cov = np.diag([    1,       1,       1,   1])*stateNoise
        # observation =              [x_m, y_m, v_m].T 
        self.obs_noise_cov = np.diag([5,        5,       2])*observationNoise

        #Initial state covariance
        self.stateCovariance = self.state_tran_noise_cov * 5
        self.observationCovariance = self.obs_noise_cov 

        self.last_observation = None

    def stateUpdate(self, dt, stateVec):
        self.state = self.getStateUpdateMatrix(dt).dot(stateVec)
        return self.state

    def getStateUpdateMatrix(self, dt): #This is equivalent to the state transition jacobean matrix
        self.stateUpdateMatrix = np.array([ [1, 0, dt, 0], \
                                            [0, 1, 0, dt], \
                                            [0, 0, 1, 0], \
                                            [0, 0, 0, 1]])
        return self.stateUpdateMatrix

    def getObservationJacobian(self, stateVec):
        '''
        state = [x,  y,  vx,  vy].T          
        '''
        vx = stateVec[2]+1e-6
        vy = stateVec[3]+1e-6
        observationJacobian = np.array(    [[1.0, 0, 0, 0], \
                                            [0, 1.0, 0, 0], \
                                            [0, 0, vx/math.sqrt(vx**2+vy**2), vx/math.sqrt(vx**2+vy**2)]])

        return observationJacobian

    def observationFunc(self, stateVec):
        '''
        state = [x,  y,  vx,  vy].T    
        observation function:
            x_m = x
            y_m = y
            v_m = sqrt(vx^2 + vy^2)      
        '''
        self.x_m = stateVec[0]
        self.y_m = stateVec[1]
        self.v_m = math.sqrt(stateVec[2]**2 + stateVec[3]**2)
        return np.array([self.x_m, self.y_m, self.v_m]).T

    def myLaps2xyv(self,obs_mylap):
        '''
        obs_mylap = [distSF_mylap, velocity], -> [x_m, y_m] 
        '''
        adjust = 1.03785
        distSF_in = obs_mylap[0]/adjust
        lateral = 7.5
        velocity = obs_mylap[1]
        
        if distSF_in < 0 :
            print('distSF_in is negative')
            distSF_in += self.length_in
        if distSF_in > self.length_in:
            print('distSF_in too large')
            distSF_in = distSF_in % self.length_in

        #Get the XY
            # enumerate through the enu look up table, find the entry with the closest distance, and interpolate the distance from 
            #point i and point i+1

        posx_in = None
        posy_in = None
        posx_out = None
        posy_out = None

        for i in range(len(self.enu_mat_in)-1): #TODO Edge cases!
            # print(i,enu[2],self.distSF_in,self.length_in)
            # if enu[2] > self.distSF_in:
            enu = self.enu_mat_in[i]
            if enu[2] >= distSF_in:
                # print(i,enu[2],distSF_in,self.length_in)
                # print('next')
                posx_in = self._map(distSF_in, self.enu_mat_in[i-1][2], enu[2],  self.enu_mat_in[i-1][0],enu[0])
                posy_in = self._map(distSF_in, self.enu_mat_in[i-1][2], enu[2],  self.enu_mat_in[i-1][1],enu[1] )
                break

        """NOTE This is a dirty hack to close the gap:"""
        if posx_in is None:
            posx_in = self._map(distSF_in, self.enu_mat_in[-3,2], self.enu_mat_in[5,2] + self.length_in,  self.enu_mat_in[-3,0],self.enu_mat_in[5,0])
        if posy_in is None:
            posy_in = self._map(distSF_in, self.enu_mat_in[-3,2], self.enu_mat_in[5,2] + self.length_in,  self.enu_mat_in[-3,1],self.enu_mat_in[5,1])
            


        distance_SF_out = distSF_in * self.in_out_ratio
        if distance_SF_out > self.length_out:
            distance_SF_out -= self.length_out
        for i, enu in enumerate(self.enu_mat_out[:-1,:]):
            if enu[2] >= distance_SF_out:
                posx_out = self._map(distance_SF_out, enu[2], self.enu_mat_out[i-1][2], enu[0], self.enu_mat_out[i-1][0])
                posy_out = self._map(distance_SF_out, enu[2], self.enu_mat_out[i-1][2], enu[1], self.enu_mat_out[i-1][1])
                break
            
        """NOTE This is a dirty hack to close the gap:"""
        if posx_out is None:
            posx_out = self._map(distance_SF_out, self.enu_mat_out[-3,2], self.enu_mat_out[5,2] + self.length_out, self.enu_mat_out[-3,0], self.enu_mat_out[5,0])
        if posy_out is None:
            posy_out = self._map(distance_SF_out, self.enu_mat_out[-3,2], self.enu_mat_out[5,2] + self.length_out, self.enu_mat_out[-3,1], self.enu_mat_out[5,1])


        self.posx_middle = self._map(lateral,0,self.track_width,posx_in,posx_out) 
        self.posy_middle = self._map(lateral,0,self.track_width,posy_in,posy_out)
        self.v_lidar   = velocity

        xyv = [self.posx_middle,self.posy_middle,velocity]
        return xyv

    def _map(self, x, in_min, in_max, out_min, out_max):
        return ((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
    
    def update(self, obs_mylap,  dt, observationCovariance=None ):
        if observationCovariance is not None:
            self.observationCovariance = observationCovariance
        '''
        obs_mylap = [distSF_mylap, velocity], -> [x_m, y_m, v_m] (by look up table)
        observation: [x,y,v]
        dt: time since last update
        '''
        if obs_mylap is None: # Give a dummy observation which is dead reckoning from the last observation
            last_distance = self.last_observation_mylap[0]
            last_velocity = self.last_observation_mylap[1]
            now_distance = last_distance + last_velocity * dt * 1.03
            obs_mylap = [now_distance, last_velocity]
            self.observationCovariance = np.diag([1,1,1])

        #Calculate the Jacobian matrix
        stateUpdateJacobian = self.getStateUpdateMatrix(dt)
        obsJacobian = self.getObservationJacobian(self.state)

        #Prediction step
        stateE = self.stateUpdate(dt,self.state)
        stateCovE = stateUpdateJacobian.dot(self.stateCovariance).dot(stateUpdateJacobian.T) + \
            self.state_tran_noise_cov

        if obs_mylap is None:
            self.state = stateE
            self.stateCovariance = stateCovE
            return self.state, self.stateCovariance, np.zeros_like(self.observationCovariance)
        else:
            # Converge the observation to xyv space after we know we have a measurement
            xyv = self.myLaps2xyv(obs_mylap)
            # adjust the data dypt of the observation
            observation = np.array(xyv).T

            #Generate Kalman Gain
            innovation = observation - self.observationFunc(self.state)
            innovationCov = obsJacobian.dot(stateCovE).dot(obsJacobian.T) + self.obs_noise_cov

            kalmanGain = stateCovE.dot(obsJacobian.T).dot(np.linalg.inv(innovationCov))
            #Correct prediction
            self.state = stateE + kalmanGain.dot(np.array(innovation))
            self.stateCovariance = (np.eye(self.stateDim) - kalmanGain.dot(obsJacobian)).dot(stateCovE)

            self.last_observation = observation
            self.last_observation_mylap = obs_mylap

            return self.state, self.stateCovariance, innovationCov


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

    mylaps_filter = MyLapsFilter(enu_in, enu_out)
    lidar_filter = ConstantVelocityFilter(stateNoise=5,observationNoise=10)


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
        env.draw(screen)

        if (time.time() - start_time)%30 > 5 :
            obs_mylap = None
            print('No mylap')
        # Use the kalman filter1
        stateVector, stateCovariance, innovationCov = mylaps_filter.update(obs_mylap, env.get_last_dt())

        # Visualize the result 1
        xpos , ypos = stateVector[0], stateVector[1]
        xpos_viz = xpos + 1920/2
        ypos_viz = ypos + 1080/2
        pygame.draw.circle(screen, (0, 0, 255), (int(xpos_viz), int(ypos_viz)), 5)
        
        # #Use the kalman filter2
        # print('no lidar')
        if obs_lidar is not None:
            obs_lidar = obs_lidar[:2]
        stateVector2 , stateCovariance2 , __ = lidar_filter.update(obs_lidar, env.get_last_dt())
        # Visualize the result 2
        xpos2 , ypos2 = stateVector2[0], stateVector2[1]
        xpos_viz2 = xpos2 + 1920/2
        ypos_viz2 = ypos2 + 1080/2
        pygame.draw.circle(screen, (0, 255, 0), (int(xpos_viz2), int(ypos_viz2)), 3)

        
        pygame.draw.aalines(screen, (255, 255, 255), True, enu_in_viz, 1)
        pygame.draw.circle(screen, (255, 255, 255), (int(enu_in_viz[0][0]),int(enu_in_viz[0][1])), 2)
        pygame.draw.aalines(screen, (255, 255, 255), True, enu_out_viz, 1)
        pygame.draw.circle(screen, (255, 255, 255), (int(enu_out_viz[0][0]),int(enu_out_viz[0][1])), 2)
        pygame.display.update()