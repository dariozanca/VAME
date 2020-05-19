import lve
import cv2
import numpy as np
import eymol # Zanca et al. 2019

#################################################
''' G-Eymol, Zanca et. al 2019 '''

class GEymol(object):

    def __init__(self, name, stimulus,
                 maximum_dimension=128.,
                 fps=25):

        # assign a name to this model
        self.name = name

        # model defintion
        self.maximum_dimension = maximum_dimension
        self.h, self.w, _ = np.shape(stimulus)
        if self.h > self.w:
            self.w_new = int((self.maximum_dimension  / self.h) * self.w)
            self.h_new = int(self.maximum_dimension )
        else:
            self.h_new = int((self.maximum_dimension  / self.w) * self.h)
            self.w_new = int(self.maximum_dimension )

        # Create an Eymol object
        parameters = {}
        parameters['fps'] = fps
        parameters['h'], parameters['w'] = self.h_new, self.w_new
        parameters['alpha_c'] = .5
        parameters['alpha_of'] = .3
        parameters['dissipation'] = 1.5
        if max(parameters['h'], parameters['w']) % 2:
            parameters['max_distance'] = max(parameters['h'], parameters['w'])
        else:
            parameters['max_distance'] = max(parameters['h'], parameters['w']) + 1
        self.foa = eymol.Eymol(parameters=parameters)

        # Simulate eye-movements for each frame
        self.parameters = parameters
        self.null_of = np.zeros((parameters['h'], parameters['w'], 2))

    def step(self, stimulus):

        stimulus_resized = cv2.resize(stimulus, (self.w_new, self.h_new), interpolation=cv2.INTER_CUBIC)

        y0_last = self.foa.next_location(stimulus_resized, self.null_of)
        y, x = y0_last[0], y0_last[1]

        # Upscale coordinates of the gaze position
        y *= float(self.h) / self.h_new # row coords.
        x *= float(self.w) / self.w_new # col coords.

        return np.array([y, x])

#################################################
''' Local G-Eymol, Faggi et al. 2020 '''

class Local_GEymol(object):

    def __init__(self, model_name, stimulus,
                 maximum_dimension=128.,
                 fps=25):

        # assign a name to this model
        self.model_name = model_name

        # model defintion
        self.maximum_dimension = maximum_dimension
        self.h, self.w = np.shape(stimulus)
        if self.h > self.w:
            self.w_new = int((self.maximum_dimension  / self.h) * self.w)
            self.h_new = int(self.maximum_dimension )
        else:
            self.h_new = int((self.maximum_dimension  / self.w) * self.h)
            self.w_new = int(self.maximum_dimension )

        # Create an object
        self.width_absorbing_layer = 0
        self.parameters = {
            "D1" : {
                'alpha_c': 1,
                'alpha_of': 1,
                'alpha_virtual': 0.0,
                'max_distance': int(0.5 * (self.w_new + self.h_new)) if int(
                  0.5 * (self.w_new + self.h_new)) % 2 == 1
                else int(0.5 * (self.w_new + self.h_new)) + 1,
                'dissipation': 10,
                'fps': fps,
                'w': self.w_new,
                'h': self.h_new,
                'y': None,
                'is_online': False,
                'alpha_fm': 0.0,
                'fixation_threshold_speed': 0.01,
                # NEW PARAMETERS
                'c': 500,
                'evaluation_method': 'implicit_method',  # or 'implicit_method'
                'width_absorbing_layer': self.width_absorbing_layer,
                'wave_absorbing_coefficient': 2000,
                'gamma': 1,
                'plot': False,
                'plot_method': 2,
                'save_plot': False,
                'epsilon': 5,
                # to evaluate the gravitational force on the foa through finite difference method
                'potential_multiplicative_factor': 50,  # to rescale the potential
                'force_multiplicative_factor': 1000000,  # to rescale the gravitational force
                'update_potential_method': 'wave_equation',
                # 'diffusion equation' or 'wave_equation'
                'theta': 1
            },

            "DW2": {
                'alpha_c': 1,
                'alpha_of': 1,
                'alpha_virtual': 0.0,
                'max_distance': int(0.5 * (self.w_new + self.h_new)) if int(
                  0.5 * (self.w_new + self.h_new)) % 2 == 1
                else int(0.5 * (self.w_new + self.h_new)) + 1,
                'dissipation': 5,
                'fps': fps,
                'w': self.w_new,
                'h': self.h_new,
                'y': None,
                'is_online': False,
                'alpha_fm': 0.0,
                'fixation_threshold_speed': 0.01,
                # NEW PARAMETERS
                'c': 1,
                'evaluation_method': 'implicit_method',  # or 'explicit_method'
                'width_absorbing_layer': self.width_absorbing_layer,
                'wave_absorbing_coefficient': 1 / 100,
                'gamma': 1 / 250000,
                'plot': False,
                'plot_method': 2,
                'save_plot': False,
                'epsilon': 5,
                # to evaluate the gravitational force on the foa through finite difference method
                'potential_multiplicative_factor': 1 / 5000,  # to rescale the potential
                'force_multiplicative_factor': 2000000,  # to rescale the gravitational force
                'update_potential_method': 'wave_equation',  # 'diffusion equation' or 'wave_equation'
                'theta': 1
            }
        }

        self.foa_processor = lve.GEymol9(self.parameters[self.model_name], I=None, V=None, device=None)

        # Simulate eye-movements for each frame
        self.null_of = lve.utils.np_float32_to_torch_float(
            np.zeros((self.parameters[self.model_name]['h'], self.parameters[self.model_name]['w'], 2))
        )

    def step(self, stimulus):

        # Prepare input stimulus
        stimulus_resized = cv2.resize(stimulus, (self.w_new, self.h_new), interpolation=cv2.INTER_CUBIC)
        stimulus_resized = lve.utils.np_uint8_to_torch_float_01(stimulus_resized)

        # Attention step
        foa, saccade = self.foa_processor.time_step(stimulus_resized, self.null_of)
        y, x = foa[0]-self.width_absorbing_layer, foa[1]-self.width_absorbing_layer

        # Upscale coordinates of the gaze position
        y *= float(self.h) / self.h_new # row coords.
        x *= float(self.w) / self.w_new # col coords.

        return np.array([y, x])