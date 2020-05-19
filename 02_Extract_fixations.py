# IMPORT EXTERNAL LIBRARIES
import numpy as np
import cv2
import os

# IMPORT SELF-DEFINED LIBRARIES
import utils

##################################################

# SIMULATED_SCANPATHS_FOLDER folder contains all the simulated scanpaths.
# It has the following structure:
# - STIMULI
#   - MODEL_NAME
#     - DATASET_1
#       - DATASET_1_SUBFOLDER
#         - STIMULUS
#           - SCANPATH_1
#           - SCANPATH_2
#           - ...
SIMULATED_SCANPATHS_FOLDER = "SIMULATED_SCANPATHS"

# Define the folder that will contain all the simulated fixations lists
SIMULATED_FIXATIONS_LISTS_FOLDER = "SIMULATED_FIXATIONS_LISTS"

# Start fixations extraction
for model_name in os.listdir(SIMULATED_SCANPATHS_FOLDER):

    for dataset in os.listdir(
            os.path.join(
                SIMULATED_SCANPATHS_FOLDER, model_name
            )
    ):

        for dataset_subfolder in os.listdir(
            os.path.join(
                SIMULATED_SCANPATHS_FOLDER, model_name, dataset
            )
        ):

            print(
                os.path.join(
                    SIMULATED_SCANPATHS_FOLDER, model_name, dataset_subfolder
                )
            )

            for stimulus_name in os.listdir(
                os.path.join(
                    SIMULATED_SCANPATHS_FOLDER, model_name, dataset, dataset_subfolder
                )
            ):
                print("-- Stimulus: ", stimulus_name)

                for simulated_scanpath_file in os.listdir(
                    os.path.join(
                        SIMULATED_SCANPATHS_FOLDER, model_name, dataset, dataset_subfolder, stimulus_name
                    )
                ):

                    simulated_scanpath_file_fullpath = os.path.join(
                        SIMULATED_SCANPATHS_FOLDER, model_name, dataset, dataset_subfolder, stimulus_name, simulated_scanpath_file
                    )

                    if os.path.isfile(simulated_scanpath_file_fullpath):

                        simulated_scanpath = utils.read_simulated_scanpath(simulated_scanpath_file_fullpath)

                        simulated_fixations_list = utils.fixations_detection(simulated_scanpath)

                        utils.save_fixations_list(
                            SIMULATED_FIXATIONS_LISTS_FOLDER,
                            model_name,
                            dataset,
                            dataset_subfolder,
                            stimulus_name,
                            simulated_scanpath_file,
                            simulated_fixations_list
                        )

                        print("-- -- Scanpath: ", simulated_scanpath_file,
                              "(", len(simulated_fixations_list), " fixations )")