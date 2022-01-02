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
    def __init__(self,enu_mat_in, enu_mat_out, distSF_in = 0,lateral = 5,velocity = 0):

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
        adjust_init = 1.01
        self.state = np.array([distSF_in, lateral, velocity, adjust_init]).T

        #Initialize the observation from my lap
        self.distSF_mylap = distSF_in * adjust_init
        self.velocity = velocity

        #Initialize the output from Lidar
        self.posx = 0
        self.posy = 0
        self.vx = 0
        self.vy = 0

        #Other parameters might be needed
        self.track_width = 15   # 15 m track width
        self.in_out_ratio = 1.03785
        self.stateDim = len(self.state)


        #Noise assumption about state transition and observation
        # state =                   [distSF_in, lateral, velocity, adjust]
        self.state_tran_noise_cov = np.diag([    1,       0.1,       1,   0.0001])
        # observation =          [distSF_mylap, velocity, x,    y,  v] 
        self.obs_noise_cov = np.diag([1,        1,       3,     3,   1])

        #Initial state covariance
        self.stateCovariance = self.state_tran_noise_cov
                                            
        self.observationCovariance = self.obs_noise_cov

    def stateUpdate(self, dt, stateVector):
        '''
        state transition:
            distSF_in = distDF_in + velocity*dt
            lateral = lateral
            velocity = velocity
            adjust = adjust
        '''
        distSF_in = stateVector[0] + stateVector[2]*dt
        lateral = stateVector[1]
        velocity = stateVector[2]
        adjust = stateVector[3]

        #Check the boundary
        if distSF_in > self.length_in:
            distSF_in -= self.length_in
        if distSF_in < 0:
            distSF_in += self.length_in

        if lateral > self.track_width:
            lateral = self.track_width
        if lateral < 0:
            lateral = 0
        
        if adjust > 1.03785:
            adjust = 1.03785
        if adjust < 1:
            adjust = 1

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
        obs_of_stateVector = self.observationFunc(stateVector)
        #Calc M1 dx/ddistSF
        state_offset = stateVector
        state_offset[0]+=0.1
        obs_diff = self.observationFunc(state_offset)-obs_of_stateVector
        m1 = obs_diff[2]*10
        #Calc M2 dx/dlateral
        state_offset = stateVector
        state_offset[1]+=0.1
        obs_diff = self.observationFunc(state_offset)-obs_of_stateVector
        m2 = obs_diff[2]*10
        #Calc M3 dy/ddistSF
        state_offset = stateVector
        state_offset[0]+=0.1
        obs_diff = self.observationFunc(state_offset)-obs_of_stateVector
        m3 = obs_diff[3]*10
        #Calc M4 dy/dlateral
        state_offset = stateVector
        state_offset[1]+=0.1
        obs_diff = self.observationFunc(state_offset)-obs_of_stateVector
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
            self.observationCovariance = observationCovariance
        '''
        observation: [x,y]
        dt: time since last update
        '''
        # adjust the data dypt of the observation
        observation = np.array(observation).T

        #Calculate the Jacobian matrix
        stateUpdateJacobian = self.getStateUpdateJacobian(dt)
        obsJacobian = self.getObservationJacobian(self.stateVector)

        #Prediction step
        stateE = self.stateUpdate(dt,self.state)
        stateCovE = stateUpdateJacobian.dot(self.stateCovariance).dot(stateUpdateJacobian.T) + \
            self.state_tran_noise_cov

        #Generate Kalman Gain
        innovation = observation - self.observationFunc(self.state)
        innovationCov = obsJacobian.dot(stateCovE).dot(obsJacobian.T) + self.obs_noise_cov

        kalmanGain = stateCovE.dot(obsJacobian.T).dot(np.linalg.inv(innovationCov))
        #Correct prediction
        self.stateVector = stateE + kalmanGain.dot(np.array(innovation))
        self.stateCovariance = (np.eye(self.stateDim) - kalmanGain.dot(obsJacobian)).dot(stateCovE)
        
        return self.stateVector, self.stateCovariance, innovationCov