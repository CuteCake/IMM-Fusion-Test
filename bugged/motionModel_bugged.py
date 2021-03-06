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
        #Generate Kalman Gain
        kalmanGain = stateCovarianceE.dot(self.observationMatrix.T).dot(np.linalg.inv(self.observationCovariance + \
            self.observationMatrix.dot(stateCovarianceE).dot(self.observationMatrix.T)))
        #Correct prediction
        self.stateVector = stateE + kalmanGain.dot(np.array(observation).T - self.observationMatrix.dot(stateE))
        self.stateCovariance = (np.eye(self.stateDim) - kalmanGain.dot(self.observationMatrix)).dot(stateCovarianceE)
        return self.stateVector

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

class TrackFollowingFilter(BaseFilter):
    '''
    state = [distSF_in, lateral, velocity, adjust].T          #(maybe add lag?)
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

    observation = 
        [distSF_mylap, velocity, x , y, v].T  

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
    def __init__(self, enu_mat_in, enu_mat_out, distSF_in = 0,lateral = 5,velocity = 0, adjust = 1.01):

        #Prepare the lookup table
        
        self.length_in = np.linalg.norm(enu_mat_in[-1,:2]-enu_mat_in[0,:2])+enu_mat_in[-1,2]#get the total length of the track
        # enu_mat_in =np.concatenate((enu_mat_in,np.array([enu_mat_in[0,0],enu_mat_in[0,1],self.length_in])[None,:]),axis=0)
        # last_enu = enu_mat_in[0,:]

        # self.enu_mat_in = np.concatenate((enu_mat_in,last_enu[None,:]),axis=0)
        self.enu_mat_in = enu_mat_in

        new_dot = np.array([enu_mat_in[0,0],enu_mat_in[0,1],self.length_in])[None,:]
        #Do it again for the outter ring

        self.length_out = np.linalg.norm(enu_mat_out[-1,:2]-enu_mat_out[0,:2])+enu_mat_out[-1,2]#get the total length of the track
        enu_mat_out =np.concatenate((enu_mat_out,np.array([enu_mat_out[0,0],enu_mat_out[0,1],self.length_out])[None,:]),axis=0) #add the first point to the end of the list
        last_enu = enu_mat_out[0,:]
        self.enu_mat_out = np.concatenate((enu_mat_out,last_enu[None,:]),axis=0)

        #Initialize the state
        self.state = np.array([distSF_in, lateral, velocity, adjust]).T


        #Initialize the output from Lidar
        self.posx = 0
        self.posy = 0
        self.vx = 0
        self.vy = 0

        #Other parameters might be needed
        self.track_width = 15   # 15 m track width
        self.in_out_ratio = self.length_out/self.length_in
        self.stateDim = len(self.state)


        #Noise assumption about state transition and observation
        # state =                   [distSF_in, lateral, velocity, adjust]
        self.state_tran_noise_cov = np.diag([    2,       0.1,       1,   0.00001])
        # observation =          [distSF_mylap, velocity, x,    y,  v] 
        self.obs_noise_cov = np.diag([5,        5,       3,     3,   1])

        #Initial state covariance
        self.stateCovariance = self.state_tran_noise_cov * 5
                                            
        self.observationCovariance = self.obs_noise_cov 

        self.last_observation = None

    def stateUpdate(self, dt, stateVector):
        '''
        state transition:
            distSF_in = distDF_in + velocity*dt
            lateral = lateral
            velocity = velocity
            adjust = adjust
        '''
        distSF_in = stateVector[0] + stateVector[2]*dt
        lateral =  stateVector[1]
        velocity = stateVector[2]
        adjust = 1.03785#stateVector[3]

        # should not Check the boundary here
        # if distSF_in > self.length_in:
        #     distSF_in -= self.length_in
        # if distSF_in < 0:
        #     distSF_in += self.length_in

        if lateral > self.track_width:
            lateral = self.track_width
        if lateral < 0:
            lateral = 0
        
        if adjust > 1.03785:
            adjust = 1.03785
        if adjust < 0.99:
            adjust = 0.99

        stateVector = np.array([distSF_in, lateral, velocity, adjust]).T
        return stateVector


    def getStateUpdateJacobian(self, dt): 
        '''
        state transition:
            distSF_in = distDF_in + velocity*dt
            lateral = lateral
            velocity = velocity
            adjust = adjust

        Funny enough, the Jacobian is not related to the state 
        '''
        Jacobian =               np.array([ [1, 0, dt, 0], \
                                            [0, 1, 0, 0], \
                                            [0, 0, 1, 0], \
                                            [0, 0, 0, 1]])
        return Jacobian
    
    def observationFunc(self,stateVector):
        '''
              state = [distSF_in, lateral, velocity, adjust].T
        observation = [distSF_mylap, velocity, x , y, v].T 
        Use the lookup table to get the observation
        '''

        distSF_in = stateVector[0].copy()
        lateral = stateVector[1]
        velocity = stateVector[2]
        adjust = stateVector[3]

        if distSF_in < 0 :
            print('distSF_in is negative')
            distSF_in += self.length_in
        if distSF_in > self.length_in:
            print('distSF_in is larger than the length of the track')
            distSF_in = distSF_in % self.length_in

        #Get the my lap observation
        self.distSF_mylap = distSF_in * adjust
        self.velocity_mylap = velocity

        #Get the Lidar observation
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
        '''
        DEBUG
        '''
        # self.test_posx_in = posx_in
        # self.test_posy_in = posy_in
        # self.test_posx_out = posx_out
        # self.test_posy_out = posy_out

        self.posx_lidar = self._map(lateral,0,self.track_width,posx_in,posx_out) 
        self.posy_lidar = self._map(lateral,0,self.track_width,posy_in,posy_out)
        self.v_lidar   = velocity

        observation = [self.distSF_mylap, velocity, self.posx_lidar,self.posy_lidar,self.v_lidar]
        return observation

    def observationFunc_np(self,stateVector):
        return np.array(self.observationFunc(stateVector)).T

    def getObservationJacobian(self,stateVector):
        '''
        state = [distSF_in, lateral, velocity, adjust]
        observation = [distSF_mylap, velocity, x,y,v] 
        observation function:
            distSF_mylap = distSF_in * adjust
            velocity_mylap = velocity         #up to 3% error, don't care
            x = from lookup table(state)
            y = from lookup table
            v = velocity

        the Jacobian matrix:
            [adjust,    0,      0,      distSF_in],
            [0,         0,      1,          0    ],
            [   M1,      M2,     0,         0    ],
            [   M3,      M4,    0,          0    ],
            [0,         0,      1,          0    ]
        '''
        obs_of_stateVector = self.observationFunc_np(stateVector)
        #Calc M1 dx/ddistSF
        state_offset = stateVector
        state_offset[0]+=0.1
        obs_diff = self.observationFunc_np(state_offset)-obs_of_stateVector
        m1 = obs_diff[2]*10
        #Calc M2 dx/dlateral
        state_offset = stateVector
        state_offset[1]+=0.1
        obs_diff = self.observationFunc_np(state_offset)-obs_of_stateVector
        m2 = obs_diff[2]*10
        #Calc M3 dy/ddistSF
        state_offset = stateVector
        state_offset[0]+=0.1
        obs_diff = self.observationFunc_np(state_offset)-obs_of_stateVector
        m3 = obs_diff[3]*10
        #Calc M4 dy/dlateral
        state_offset = stateVector
        state_offset[1]+=0.1
        obs_diff = self.observationFunc_np(state_offset)-obs_of_stateVector
        m4 = obs_diff[3]*10

        adjust = stateVector[3]
        distSF_in = stateVector[0]

        #assimble the Jacobian matrix
        JacobianMatrix = np.array([[adjust, 0,       0, distSF_in], \
                                    [0,     0,      1,      0], \
                                    [m1,    m2,     0,     0], \
                                    [m3,    m4,     0,      0], \
                                    [0,     0,      1,      0]])
        return JacobianMatrix

    def _map(self, x, in_min, in_max, out_min, out_max):
        return ((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

    def update(self, observation,  dt, observationCovariance=None ):
        if observationCovariance is not None:
            self.obs_noise_cov = observationCovariance
        '''
        observation =  [distSF_mylap, velocity, x , y, v].T  
        dt: time since last update
        '''
        # adjust the data dypt of the observation
        observation = np.array(observation).T
        

        #Calculate the Jacobian matrix
        stateUpdateJacobian = self.getStateUpdateJacobian(dt)
        obsJacobian = self.getObservationJacobian(self.state)

        #Prediction step
        stateE = self.stateUpdate(dt,self.state)
        stateCovE = stateUpdateJacobian.dot(self.stateCovariance).dot(stateUpdateJacobian.T) + \
            self.state_tran_noise_cov

        #Generate Kalman Gain
        innovation = observation - self.observationFunc(self.state)
        innovationCov = obsJacobian.dot(stateCovE).dot(obsJacobian.T) + self.obs_noise_cov

        kalmanGain = stateCovE.dot(obsJacobian.T).dot(np.linalg.inv(innovationCov))
        #Correct prediction
        self.state = stateE + kalmanGain.dot(np.array(innovation))
        self.stateCovariance = (np.eye(self.stateDim) - kalmanGain.dot(obsJacobian)).dot(stateCovE)

        # distSF_in = self.state[0] + self.state[2]*dt
        # lateral = self.state[1]
        # velocity = self.state[2]
        # adjust = self.state[3]

        # #Check the boundary
        # if distSF_in > self.length_in:
        #     distSF_in -= self.length_in
        # if distSF_in < 0:
        #     distSF_in += self.length_in

        # if lateral > self.track_width:
        #     lateral = self.track_width
        # if lateral < 0:
        #     lateral = 0
        
        # if adjust > 1.03785:
        #     adjust = 1.03785
        # if adjust < 0.99:
        #     adjust = 0.99

        # self.state = np.array([distSF_in, lateral, velocity, adjust]).T


        self.last_observation = observation

        return self.state, self.stateCovariance, innovationCov

    def update_mylap_only(self, observation,  dt, observationCovariance=None ):
        '''
        observation = [distSF_mylap, velocity].T  
        dt: time since last update
        '''
        # distSF_in = self.state[0] + self.state[2]*dt
        lateral = self.state[1]
        # velocity = self.state[2]
        adjust = self.state[3]

        # self.substate = np.array([distSF_in, velocity]).T
        # adjust the data dypt of the observation
        observation = np.array(observation).T

        #Calculate the Jacobian matrix
        stateUpdateJacobian = self.getStateUpdateJacobian(dt)
        obsJacobian = self.getObservationJacobian(self.state)

        """
        The different part: the observation and the jacobian matrix is different
        """
        observation = observation[:2]
        obsJacobian = obsJacobian[:2,:]

        #Prediction step
        stateE = self.stateUpdate(dt,self.state)
        stateCovE = stateUpdateJacobian.dot(self.stateCovariance).dot(stateUpdateJacobian.T) + \
            self.state_tran_noise_cov

        #Generate Kalman Gain
        innovation = observation - self.observationFunc(self.state)[0:2]
        ''' The different part: obs func and obs cov is different'''
        innovationCov = obsJacobian.dot(stateCovE).dot(obsJacobian.T) + self.obs_noise_cov[:2,:2]

        kalmanGain = stateCovE.dot(obsJacobian.T).dot(np.linalg.inv(innovationCov))
        #Correct prediction
        self.state = stateE + kalmanGain.dot(np.array(innovation))
        self.stateCovariance = (np.eye(self.stateDim) - kalmanGain.dot(obsJacobian)).dot(stateCovE)

        # going back check
        if self.last_observation is not None:
            if self.last_observation[0] > observation[0]+100:  #the car passing start line
                self.state[0] = observation[0]
                adjust = self.state[3]

        if lateral > self.track_width:
            lateral = self.track_width
        if lateral < 0:
            lateral = 0
        
        if adjust > 1.03785:
            adjust = 1.03785
        if adjust < 0.99:
            adjust = 0.99

        if self.state[0] < 0 or self.state[0] > self.length_in:
            self.state = stateE
        
        #By passing the EKF
        self.state[1] = lateral
        self.state[3] = adjust

        self.last_observation = observation

        return self.state, self.stateCovariance, innovationCov


    def getXY(self,stateVector):
        '''
        state = [distSF_in, lateral, velocity, adjust]
        obs = [distSF_mylap, velocity, x , y, v].T 
        '''
        obs_of_stateVector = self.observationFunc(stateVector)
        return obs_of_stateVector[2],obs_of_stateVector[3]

if __name__ == '__main__':

    from enviroment_bugged import *
    import time
    import matplotlib.pyplot as plt
    enu_in = prepare_data('vegas_insideBounds.csv')
    enu_out = prepare_data('vegas_outsideBounds.csv')

    #initialize visualization 
    enu_in_viz = enu_in[:,:2] + np.array([1920/2,1080/2])
    enu_out_viz = enu_out[:,:2] + np.array([1920/2,1080/2])
    
    
    env = PointEnvRaceTrack(1920, 1080, enu_in, enu_out, distSF_in=2000, lateral=10, velocity=100,adjust=1.03785)

    track_filter = TrackFollowingFilter(enu_in, enu_out, distSF_in=200, lateral= 5,velocity=10, adjust=1.03785)
    
    # #plot a line of x calculated from distSF_in
    # x_in = np.linspace(0,track_filter.length_in+100,100)
    # test_posx_in = []
    # test_posy_in = []
    # test_posx_out = []
    # test_posy_out = []
    # for x in x_in:
    #     track_filter.getXY(np.array([x,0,0,1.0]))
    #     test_posx_in.append(track_filter.test_posx_in)
    #     test_posy_in.append(track_filter.test_posy_in)
    #     test_posx_out.append(track_filter.test_posx_out)
    #     test_posy_out.append(track_filter.test_posy_out)
    
    # #plot 4 subplots for each value
    # fig, axs = plt.subplots(4,1,figsize=(10,10))
    # axs[0].plot(x_in,test_posx_in)
    # axs[0].set_title('test_posx_in')
    # axs[2].plot(x_in,test_posx_out)
    # axs[2].set_title('test_posx_out')
    # axs[1].plot(x_in,test_posy_in)
    # axs[1].set_title('test_posy_in')
    # axs[3].plot(x_in,test_posy_out)
    # axs[3].set_title('test_posy_out')
    # plt.show()


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

        #concatenate the observation
        obs = obs_mylap.copy()
        for i in obs_lidar:
            obs.append(i)

        # Use the kalman filter
        if (time.time() - start_time)%10 > 5 :
            print('using Lidar Fusion')
            stateVector, stateCovariance, innovationCov = track_filter.update(obs, env.get_last_dt())
        

        # if (time.time() - start_time)%10 > 5:
        #     obs_mylap = None
        #     print("dropping mylap")
        
        stateVector, stateCovariance, innovationCov = track_filter.update_mylap_only(obs_mylap, env.get_last_dt())

        # print(stateVector,stateCovariance[0][0])
        xpos , ypos = track_filter.getXY(stateVector)

        xpos_viz = xpos + 1920/2
        ypos_viz = ypos + 1080/2
        
        pygame.draw.circle(screen, (0, 0, 255), (int(xpos_viz), int(ypos_viz)), 5)
        
        pygame.draw.aalines(screen, (255, 255, 255), True, enu_in_viz, 1)
        pygame.draw.circle(screen, (255, 255, 255), (int(enu_in_viz[0][0]),int(enu_in_viz[0][1])), 2)
        pygame.draw.aalines(screen, (255, 255, 255), True, enu_out_viz, 1)
        pygame.draw.circle(screen, (255, 255, 255), (int(enu_out_viz[0][0]),int(enu_out_viz[0][1])), 2)
        pygame.display.update()