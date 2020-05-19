# IMPORT EXTERNAL LIBRARIES
import numpy as np
import cv2
import os

# IMPORT SELF-DEFINED LIBRARIES
import utils

##################################################

# Here we indicate the name of the folder containing all the stimuli
STIMULI_FOLDER = "STIMULI"

# Here we indicate which is the folder containing all the simulated fixations lists
SIMULATED_FIXATIONS_LISTS_FOLDER = "SIMULATED_FIXATIONS_LISTS"

# SIMULATED_FIXMAPS_FOLDER folder contains all the simulated Fixations Maps.
# It has the following structure:
# - STIMULI
#   - MODEL_NAME
#     - DATASET_1
#       - DATASET_1_SUBFOLDER
#         - STIMULUS
#           - Fixations map
SIMULATED_FIXMAPS_FOLDER = "SIMULATED_FIXMAPS"

# Start fixations extraction
for model_name in os.listdir(SIMULATED_FIXATIONS_LISTS_FOLDER):

    for dataset in os.listdir(
            os.path.join(
                SIMULATED_FIXATIONS_LISTS_FOLDER, model_name
            )
    ):

        for dataset_subfolder in os.listdir(
            os.path.join(
                SIMULATED_FIXATIONS_LISTS_FOLDER, model_name, dataset
            )
        ):

            print(
                os.path.join(
                    SIMULATED_FIXATIONS_LISTS_FOLDER, model_name, dataset_subfolder
                )
            )

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
                    stimulus_name, _ = os.path.splitext(stimulus_filename)
                    stimulus = cv2.imread(
                        os.path.join(
                            STIMULI_FOLDER, dataset, dataset_subfolder, stimulus_filename
                        ), 0
                    )
                    print("-- Stimulus: ", stimulus_name)
                    path_to_fixations_list_folder = os.path.join(
                        SIMULATED_FIXATIONS_LISTS_FOLDER, model_name, dataset, dataset_subfolder, stimulus_name
                    )
                    utils.save_fixmap(
                        SIMULATED_FIXMAPS_FOLDER,
                        path_to_fixations_list_folder,
                        model_name,
                        dataset,
                        dataset_subfolder,
                        stimulus_name,
                        stimulus
                    )