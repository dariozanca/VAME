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

##################################################

# Here we indicate the name of the folder containing all the stimuli
STIMULI_FOLDER = "STIMULI"

# Here we indicate the folder containing all the human fixations maps
HUMAN_FIXMAPS_FOLDER = "HUMAN_FIXMAPS"

# Here we indicate which is the folder containing all the simulated fixations lists
SIMULATED_FIXATIONS_LISTS_FOLDER = "SIMULATED_FIXATIONS_LISTS"

# Here we indicate the folder containing all the simulated fixations maps
# Notice: they are saved as compressed sparse numpy matrix
SIMULATED_FIXMAPS_FOLDER = "SIMULATED_FIXMAPS"

# Here we define and initialize the CSV file which will contain all the saliency scores
csv_file = open('SaliencyScores.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
# Save score
csv_writer.writerow(
    [
        "Model name",
        "Dataset",
        "Category",
        "Stimulus name",
        "Blur",
        "Center bias",
        "AUC-JUDD",
        "NSS"
    ]
)

# Start fixations extraction
for model_name in os.listdir(
        SIMULATED_FIXMAPS_FOLDER
):

    for dataset in os.listdir(
            os.path.join(
                SIMULATED_FIXMAPS_FOLDER, model_name
            )
    ):

        for dataset_subfolder in os.listdir(
            os.path.join(
                SIMULATED_FIXMAPS_FOLDER, model_name, dataset
            )
        ):

            print(
                os.path.join(
                    SIMULATED_FIXMAPS_FOLDER, model_name, dataset_subfolder
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

                    simulated_fixmap = scipy.sparse.load_npz(
                        os.path.join(
                            SIMULATED_FIXMAPS_FOLDER,
                            model_name,
                            dataset,
                            dataset_subfolder,
                            stimulus_name,
                            'FixMap.npz'
                        )
                    ).todense()

                    human_fixmap = sio.loadmat(
                        os.path.join(
                            HUMAN_FIXMAPS_FOLDER,
                            dataset,
                            dataset_subfolder,
                            stimulus_name + '.mat'
                        )
                    )['fixLocs']


                    # Load center matrix
                    h, w = np.shape(simulated_fixmap)
                    center_matrix = sio.loadmat('center.mat')['center']
                    center_matrix = cv2.resize(center_matrix, (w, h))
                    if not center_matrix.max() == 0:
                        center_matrix /= center_matrix.max()

                    for Blur in (60, 80, 100):
                        for Center in (.2, .5, .8):

                            # Apply blurring
                            simulated_saliency_map = cv2.GaussianBlur(
                                simulated_fixmap.astype(float),
                                (10 * Blur - 1, 10 * Blur - 1),
                                Blur - 1
                            )
                            if not (simulated_saliency_map.max() == 0):
                                simulated_saliency_map /= simulated_saliency_map.max()

                            # Apply center bias
                            simulated_saliency_map = (1. - Center) * simulated_saliency_map + Center * center_matrix

                            # Compute saliency metrics
                            nss = FixaTons.metrics.NSS(simulated_saliency_map, human_fixmap)
                            auc = FixaTons.metrics.AUC_Judd(simulated_saliency_map, human_fixmap)

                            # Save score
                            csv_writer.writerow(
                                [
                                    model_name,
                                    dataset,
                                    dataset_subfolder,
                                    stimulus_name,
                                    Blur,
                                    Center,
                                    auc,
                                    nss
                                ]
                            )

                            # Print some informations
                            print(
                                "-- -- AUC-JUDD:", auc,
                                "  NSS:", nss,
                                "  ( BLUR:", Blur,
                                "  CENTER:", Center, ")"
                            )
                    print("-- -- -- Total execution time:", time.time() - start_time, "s")