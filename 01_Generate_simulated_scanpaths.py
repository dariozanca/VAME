# IMPORT EXTERNAL LIBRARIES
import numpy as np
import cv2
import os
from multiprocessing import Process, Lock
import time

# IMPORT SELF-DEFINED LIBRARIES
import utils


##################################################

# STIMULI folder contains all the stimuli on which all attention models
# will be evaluated. It has the following structure:
# - STIMULI
#   - DATASET_1
#     - DATASET_1_SUBFOLDER
#       - img_1
#       - img_2
#       - ...
STIMULI_FOLDER = "STIMULI"
NUMBER_OF_SIMULATED_SCANPATHS = 5

# Define the solder in which all simulated scanpath are to be saved
SIMULATED_SCANPATHS_FOLDER = "SIMULATED_SCANPATHS"

# Define the list of models to be evaluated. Please notice:
# Models must be defined in models.py
MODELS_LIST = ["D1", "DW2"]

# Start simulations
if __name__ == '__main__':

    #for dataset in os.listdir(STIMULI_FOLDER):
    for dataset in ('FixaTons', ):

        for dataset_subfolder in os.listdir(
            os.path.join(
                STIMULI_FOLDER, dataset
            )
        ):
            print(os.path.join(STIMULI_FOLDER, dataset, dataset_subfolder))
            for stimulus_filename in os.listdir(
                os.path.join(
                    STIMULI_FOLDER, dataset, dataset_subfolder
                )
            ):
                if os.path.isfile(
                        os.path.join(
                            STIMULI_FOLDER, dataset, dataset_subfolder, stimulus_filename
                        )
                ):
                    print("-- ", stimulus_filename)
                    start_time = time.time() # start simulations on a single stimulus
                    stimulus = cv2.imread(
                        os.path.join(
                            STIMULI_FOLDER, dataset, dataset_subfolder, stimulus_filename
                        ), 0
                    )
                    for model_name in MODELS_LIST:
                        lock = Lock()
                        ps = []
                        for simulated_scanpath_ID in range(NUMBER_OF_SIMULATED_SCANPATHS):
                            ps.append(
                                Process(target=utils.scanpath_single_process,
                                    args=(
                                        lock,
                                        simulated_scanpath_ID,
                                        model_name,
                                        stimulus,
                                        SIMULATED_SCANPATHS_FOLDER,
                                        dataset,
                                        dataset_subfolder,
                                        stimulus_filename
                                    )
                                )
                            )
                            ps[-1].start()

                        for simulated_scanpath_ID in range(NUMBER_OF_SIMULATED_SCANPATHS):
                            ps[simulated_scanpath_ID].join()

                        print("-- -- -- Total execution time:", time.time() - start_time, "s")

