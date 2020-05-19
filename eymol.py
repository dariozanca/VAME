'''
Created on:	29 May 2017

@author: 	Eymol Zanca (dario.zanca@unifi.it, dariozanca@gmail.com)

            Ph.D. Student in Smart Computing,
                University of Florence and University of Siena.

@summary: 	Collection of functions to generate scanpaths with EYMOL.
'''

#########################################################################################

# IMPORT EXTERNAL LIBRARIES

import numpy as np
import cv2
from math import sin, pi, isnan
from random import randint, uniform
from scipy.integrate import odeint
import time

########################################################################################################################
########################################################################################################################

''' 
Main class to create an istance of the model.

Example of use:

    params = {'alpha_c': 0.1, 'alpha_of': 0.2, 'max_distance': 300}
      
    foa = Eymol(params)
    
    for t in range(T):
        foa.next_location(frame_t, of_t)        
'''

class Eymol():

    def __init__(self, parameters):

        ''' parameters: it is a dictionary of parameters.
                'alpha_c': 	        weight for gradient input,
                                    suggested value 0.1
                'alpha_of':         weight for optical flow input,
                                    suggested value 0.2
                'max_distance':     maximum distance from actual point to consider in the integral
                                    suggested value average image dimensions
                'dissipation':      weigth of the term of dissipation
                                    suggested value 0.1
                'frame_rate':       frame per second of the input video stream

                'h_w':              frame size list

                'is_online':        True if you argoing with webcam, False otherwise
        '''

        # Initial state
        self.t = 0
        self.y = []

        # Parameters
        self.parameters = parameters

        max_d = parameters['max_distance']

        ### self.is_online = parameters['is_online']
        self.is_online = False 

        self.frame_rate = parameters['fps']
        self.h, self.w = parameters['h'], parameters['w']

        self.saccades_per_second = 3.
        self.real_time_last_saccade = time.clock()

        # Generate distances matrix
        self.distances_matrix = create_distances_matrix( max_d )

        # Generate a matrix to mark pixel to which inhibit return
        self.IOR_matrix = np.zeros( (self.h, self.w) )


    def next_location(self, frame_t, of_t):

        '''
            Input:
                frame_t: RGB image
                of_t: optical flow (2 channels)

            Output:
                y = [row, column, row velocity, column velocity] of the next location
        '''

        gradient_t = get_gradients(frame_t)

        self.y = compute_next_location(
                            # Visual input
                            feature_maps = (gradient_t, of_t),

                            # Initial condition of the system and time instants to integrate
                            y0 = self.y,
                            times = np.arange(self.t, self.t + 1, .1),

                            # System parameters
                            parameters = self.parameters,

                            distances_matrix = self.distances_matrix,

                            IOR_matrix = self.IOR_matrix
                            )

        self.t += 1


        # TODO: pezza momentanea
        # restituisci solo pixel dentro il frame
        y_out = self.y
        y_out[0], y_out[1] = stayinside(frame_t, row_col=y_out[0:2])

        # add pixel to the inhibition of return matrix
        if not self.is_online:
            if self.t % int(self.frame_rate / self.saccades_per_second) == 0:
                self.IOR_matrix = inhibit_return_in(self.IOR_matrix, row_col=y_out[0:2])

        else:
            if time.clock() - self.real_time_last_saccade >= (1. / self.saccades_per_second):
                self.IOR_matrix = inhibit_return_in(self.IOR_matrix, row_col=y_out[0:2])
                self.real_time_last_saccade = time.clock() # update real time of the last saccade

        return y_out

    def reset(self, y=[]):

        # Initial state
        self.t = 0
        self.y = y

########################################################################################################################
########################################################################################################################

def compute_next_location(
                            # Visual input
                            feature_maps,

                            # Initial condition of the system and 
                            # time instants to integrate
                            times,
                            y0,

                            # System parameters
                            parameters,

                            distances_matrix,

                            IOR_matrix
                            ):

    ''' Given input feature maps, this function returns the next location of the visual
        attention scanpath '''

    "Get feature maps dimensions"
    h, w, _ = feature_maps[0].shape

    "Add parameters"
    parameters['k'] = 10**6
    r = 0
    parameters['r'] = (r, h - r, r, w - r)


    "Numerical method"

    # If not provided, generate random initial conditions
    if not y0: y0 = generate_initial_conditions(h,w)

    # Generate scanpath (by integrating diff. equations)
    y = odeint(myode, y0, times,
               args=(feature_maps, parameters, distances_matrix, IOR_matrix),
               mxstep=100, rtol=.1, atol=.1
               )

    return list(y[-1])

########################################################################################################################

def generate_initial_conditions(h,w):

    ''' This function generates initial condition for the dynamical system to be
    integrated. Numbers used here are arbitrary. Consider to motify or determine better
    numbers in future implementations. '''

    initRay = int(min(h, w) * 0.17)
    x1_init = int(h / 2) + randint(-initRay, initRay)
    x2_init = int(w / 2) + randint(-initRay, initRay)
    v1_init = 2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))
    v2_init = 2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))

    return [x1_init, x2_init, v1_init, v2_init]

########################################################################################################################

def crop(frame, x_y, n):

    x, y = x_y

    if n % 2:
        d = (n//2)
    else:
        d = (n // 2) + 1

    h, w = np.shape(frame)

    if x < 0: x = 0
    elif x >= h: x = h-1

    if y < 0: y = 0
    elif y >= w: y = w-1

    x = int(x) + d
    y = int(y) + d

    frame = cv2.copyMakeBorder(frame,d,d,d,d,cv2.BORDER_CONSTANT,value=0)

    return frame[x-d:x+d+1, y-d:y+d+1]

########################################################################################################################

def myode(y, t, feature_maps, parameters, distances_matrix, IOR_matrix):

    '''	This function describes the system of two second-order differential
        equations which describe visual attention. (VERSION 3 - GRAVITATIONAL)

        y: it is the vector of the variables (x1, x2, dot x1, dot x2)

        t: time (frames)

        parameters: dictionary containing all the parameters of the model '''

    # Get parameters
    T = parameters['fps']
    dissipation = parameters['dissipation']
    alpha_c = parameters['alpha_c']
    alpha_of = parameters['alpha_of']
    k = parameters['k']
    r1_1, r1_2, r2_1, r2_2 = parameters['r']

    # curiosity feature map
    grandient_t = feature_maps[0]
    if not grandient_t.max() == 0: grandient_t /= grandient_t.max()
    curiosity = np.sqrt(grandient_t[:,:,0]**2 + grandient_t[:,:,1]**2)

    # Apply IOR function (Inhibition of Return)
    curiosity *= (1-IOR_matrix)

    # optical flow feature map
    of_t = feature_maps[1].astype(float)
    if not of_t.max() == 0: of_t /= of_t.max()
    opticalflow = np.sqrt(of_t[:,:,0]**2 + of_t[:,:,1]**2)

    # get outliers (it solves egomotion) TODO: talk about it in the report
    opticalflow -= opticalflow.mean()
    opticalflow = np.abs(opticalflow)

    # Apply distances matrix

    n = np.shape(distances_matrix)[1]

    curiosity_crop = crop(curiosity, (y[0], y[1]), n)
    if not curiosity_crop.max() == 0: curiosity_crop /= curiosity_crop.max() 

    opticalflow_crop = crop(opticalflow, (y[0], y[1]), n)
    if not opticalflow_crop.max() == 0: opticalflow_crop /= opticalflow_crop.max()

    # define gravitational fields contributions

    C_x = alpha_c   *   np.array(
                   [   (distances_matrix[0, :, :] * curiosity_crop).sum(),

                       (distances_matrix[1, :, :] * curiosity_crop).sum()    ]   )

    OF_x = alpha_of  *  np.array(
                   [   (distances_matrix[0, :, :] * opticalflow_crop).sum(),

                       (distances_matrix[1, :, :] * opticalflow_crop).sum()    ]   )

    "System of differential equations"

    dy = [  y[2],

            y[3],

            C_x[0] + OF_x[0] - dissipation*y[2],

            C_x[1] + OF_x[1] - dissipation*y[3]
          ]

    return dy

########################################################################################################################

def create_distances_matrix(n):

    ''' Create distances_mask for sum on the frame
        (x - a) / |x-a|**2
        notice: (x-a) is a vector.
        The resulting matrix is of dimension 2 x w x h. '''

    distances_matrix = np.zeros((2, n, n))

    center_x, center_y = (n//2), (n//2)

    for i in range(n):
        for j in range(n):
            if not (i == center_x and j == center_y):
                distances_matrix[0, i, j] = (n//10 + 1) * float(i - center_x) / (
                        ((i-center_x)**2 + (j - center_y)**2) + (n//10))

    for i in range(n):
        for j in range(n):
            if not (i == center_x and j == center_y):
                distances_matrix[1, i, j] = (n//10 + 1) * float(j - center_y) / (
                        ((i-center_x)**2 + (j - center_y)**2)  + (n//10))

    return distances_matrix


########################################################################################################################

def write_red_dot(frame, row_col,
                  RAY=5,
                  fixation_flag=False,
                  col_fix=(255, 0, 0),
                  col_sac=(0, 0, 255)):

    row, col = row_col

    # get point coordinates
    if isnan(row) or isnan(col):
        row, col = 0, 0
    else:
        row, col = int(row), int(col)

    if (row - RAY < 0):
        row = RAY
    else:
        if (row + RAY >= np.shape(frame)[0]):
            row = np.shape(frame)[0] - RAY - 1
    if (col - RAY < 0):
        col = RAY
    else:
        if (col + RAY >= np.shape(frame)[1]):
            col = np.shape(frame)[1] - RAY - 1

    if fixation_flag:
        cv2.circle(frame,
                   (col, row),
                   RAY, col_fix, 1)
    else:
        cv2.circle(frame,
                   (col, row),
                   RAY, col_sac, -1)

    return frame

########################################################################################################################

def gaussian(frame, row_col, RAY=25, blur=151):

    ''' This function returns a new frame with the same dimensions of frame, with a gaussian centered in the
        position (row, col).
        For a fast implementation, the gaussian is draw as a circle and then gaussian blurring is applied.  '''

    row, col = row_col[0], row_col[1]
    new_frame = np.zeros(np.shape(frame))
    cv2.circle(new_frame,
               (col, row),
               RAY, (1,), -1)
    new_frame = cv2.GaussianBlur(new_frame,(blur,blur),0)
    if not new_frame.max() == 0: new_frame /= new_frame.max()
    return new_frame

def inhibit_return_in(frame, row_col, RAY=5):

    row, col = stayinside(frame, row_col, RAY=RAY)

    new_frame = gaussian(frame, (row, col), RAY=RAY)

    frame = 0.9 * frame

    # add new inhibition signal
    frame += new_frame

    # Cut values greater than 1
    frame[frame>1] = 1.

    return frame


########################################################################################################################

def stayinside(frame, row_col, RAY=5):

    row, col = row_col

    # get point coordinates
    if isnan(row) or isnan(col):
        row, col = 0, 0
    else:
        row, col = int(row), int(col)

    if (row - RAY < 0):
        row = RAY
    else:
        if (row + RAY >= np.shape(frame)[0]):
            row = np.shape(frame)[0] - RAY - 1
    if (col - RAY < 0):
        col = RAY
    else:
        if (col + RAY >= np.shape(frame)[1]):
            col = np.shape(frame)[1] - RAY - 1

    return row, col

########################################################################################################################

def get_gradients(frame_t):

    sobelx = cv2.Sobel(frame_t, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame_t, cv2.CV_64F, 0, 1, ksize=5)

    return np.dstack( (sobelx, sobely) )

########################################################################################################################

def euclidean_distance(x,y):

    sum = 0

    for i in range(len(x)):

        sum += (x[i] - y[i])**2

    return sum**.5

def wave(frame, t, T=25):

    ''' n: dimension of the squared frame
        T: period of the wave (in frames) '''

    # this is to have a complete period in "frame_rate" number of frames
    omega = (2 * pi) / T

    # get dimensions
    h, w = np.shape(frame)[0], np.shape(frame)[1]

    # get some parameters that depend on the image
    C = h//2, w//2 # center of the image
    L = euclidean_distance(C, (0,0)) # maximum distance from the center of the image

    # compute the wave function
    wave = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            wave[i, j] = sin(omega*t + (pi/2)*(euclidean_distance(C, (i,j))/L))**2

    return wave

def create_wave_matrix(h_w, T):

    ''' (h, w): dimensions of the frame
        T: period of the wave (in frames) '''

    h, w = h_w

    T = int(T)  # fix, hack?
    wave_batch = np.zeros((T,h,w))

    for t in range(T):

        wave_batch[t] = wave(wave_batch[t], t, T)

    return wave_batch

########################################################################################################################
