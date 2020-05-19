# IMPORT EXTERNAL LIBRARIES
import numpy as np
import cv2
import os
import scipy
import scipy.io as sio
import csv
import time

# IMPORT SELF-DEFINED LIBRARIES
import FixaTons

# IMPORT SELF-DEFINED LIBRARIES
import utils

##################################################

# Here we indicate the name of the folder containing all the stimuli
STIMULI_FOLDER = "STIMULI"

# Here we indicate the folder containing all the human scanpaths
HUMAN_SCANPATHS_FOLDER = "HUMAN_SCANPATHS"

# Here we indicate which is the folder containing all the simulated fixations lists
SIMULATED_FIXATIONS_LISTS_FOLDER = "SIMULATED_FIXATIONS_LISTS"


# Here we define and initialize the CSV file which will contain all the saliency scores
csv_file = open('ScanpathScores.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
# Save score
csv_writer.writerow(
    [
        "Model name",
        "Dataset",
        "Category",
        "Stimulus name",
        "SED", # String-edit distance (distance)
        "STDE" # Scaled time-delay embedding (similarity)
    ]
)

# Start fixations extraction
for model_name in os.listdir(
        SIMULATED_FIXATIONS_LISTS_FOLDER
):

    # for dataset in os.listdir(
    #         os.path.join(
    #             SIMULATED_FIXATIONS_LISTS_FOLDER, model_name
    #         )
    # ):
    for dataset in ('FixaTons', ):

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
                    start_time = time.time() # start simulations on a single stimulus
                    stimulus_name, _ = os.path.splitext(stimulus_filename)
                    stimulus = cv2.imread(
                        os.path.join(
                            STIMULI_FOLDER, dataset, dataset_subfolder, stimulus_filename
                        ), 0
                    )
                    print("-- Stimulus: ", stimulus_name)

                    for simulated_fixations_list_file in os.listdir(
                            os.path.join(
                                SIMULATED_FIXATIONS_LISTS_FOLDER,
                                dataset,
                                dataset_subfolder,
                                stimulus_filename
                            )
                    ):

                        simulated_fixations_list_file_fullpath = os.path.join(
                            os.path.join(
                                SIMULATED_FIXATIONS_LISTS_FOLDER,
                                dataset,
                                dataset_subfolder,
                                stimulus_filename,
                                simulated_fixations_list_file
                            )
                        )

                        if os.path.isfile(simulated_fixations_list_file_fullpath):

                            simulated_fixations_list = utils.read_simulated_scanpath(
                                simulated_fixations_list_file_fullpath
                            )[:, (0, 1)] # row col

                            for human_fixations_list_file in os.listdir(
                                    os.path.join(
                                        HUMAN_SCANPATHS_FOLDER,
                                        dataset,
                                        dataset_subfolder,
                                        stimulus_filename
                                    )
                            ):

                                human_fixations_list_file_fullpath = os.path.join(
                                    os.path.join(
                                        HUMAN_SCANPATHS_FOLDER,
                                        dataset,
                                        dataset_subfolder,
                                        stimulus_filename,
                                        human_fixations_list_file
                                    )
                                )

                                if os.path.isfile(human_fixations_list_file_fullpath):
                                    human_fixations_list = utils.read_simulated_scanpath(
                                        human_fixations_list_file_fullpath
                                    )[:, (1,0)] # row col is inverted in FixaTons

                                    # Compute scanpath metrics
                                    sed = FixaTons.metrics.string_edit_distance(
                                        stimulus,
                                        human_fixations_list,
                                        simulated_fixations_list
                                    )
                                    stde = FixaTons.metrics.scaled_time_delay_embedding_similarity(
                                        human_fixations_list,
                                        simulated_fixations_list,
                                        stimulus
                                    )

                                    # Save score
                                    csv_writer.writerow(
                                        [
                                            model_name,
                                            dataset,
                                            dataset_subfolder,
                                            stimulus_name,
                                            sed,
                                            stde
                                        ]
                                    )

                                    # Print some informations
                                    print(
                                        "-- -- String-edit distance", sed,
                                        "  Scaled Time-delay Embeddings:", stde
                                    )
                            print("-- -- -- Total execution time:", time.time() - start_time, "s")