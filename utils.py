# IMPORT EXTERNAL LIBRARIES
import numpy as np
import cv2
import os
import csv
import scipy

# IMPORT SELF-DEFINED LIBRARIES
import detectors
from models import Local_GEymol

#################################################

### 01


def simulated_scanpath(attention_model, stimulus, seconds=5, fps=25, max_dim_px = 224.):

    # Simulate eye-movements for each frame
    number_of_frames = seconds*fps
    scanpath = np.zeros((number_of_frames, 3))
    for t in range(number_of_frames):
        y0_last = attention_model.step(stimulus)
        scanpath[t, 0], scanpath[t, 1], scanpath[t, 2] = y0_last[0], y0_last[1], float(t)/fps

    return scanpath


def save_scanpath(
        SIMULATED_SCANPATHS_FOLDER,
        model_name,
        dataset,
        dataset_subfolder,
        stimulus_filename,
        simulated_scanpath_ID,
        simulated_scanpath
):

    # Strip extension of the file from the filename
    stimulus_name, _ = os.path.splitext(stimulus_filename)

    # define saving path (create folder in case they do not exist still)
    path = os.path.join(SIMULATED_SCANPATHS_FOLDER)
    if not os.path.exists(path): os.mkdir(path)
    path = os.path.join(path, model_name)
    if not os.path.exists(path): os.mkdir(path)
    path = os.path.join(path, dataset)
    if not os.path.exists(path): os.mkdir(path)
    path = os.path.join(path, dataset_subfolder)
    if not os.path.exists(path): os.mkdir(path)
    path = os.path.join(path, stimulus_name)
    if not os.path.exists(path): os.mkdir(path)

    # save the array as file
    np.savetxt(
        os.path.join(
            path, str(simulated_scanpath_ID)
        ), simulated_scanpath
    )

def scanpath_single_process(
        l,
        simulated_scanpath_ID,
        model_name,
        stimulus,
        SIMULATED_SCANPATHS_FOLDER,
        dataset,
        dataset_subfolder,
        stimulus_filename
):
    # Print scanpath state to the std output
    l.acquire()
    print("-- -- Scanpath:", simulated_scanpath_ID, "(", model_name, ")")
    l.release()

    m = Local_GEymol(model_name, stimulus)

    s = simulated_scanpath(m, stimulus)

    l.acquire()
    save_scanpath(
        SIMULATED_SCANPATHS_FOLDER,
        m.model_name,
        dataset,
        dataset_subfolder,
        stimulus_filename,
        simulated_scanpath_ID,
        s
    )
    l.release()


### 02


def read_simulated_scanpath(simulated_scanpath_path):

    with open(simulated_scanpath_path, 'r') as f:
        f_reader = csv.reader(f, delimiter=' ')
        simulated_scanpath = []
        for line in f_reader:
            simulated_scanpath.append(np.array(line).astype(float))

    return np.array(simulated_scanpath)


def fixations_detection(simulated_scanpath):

    # _, simulated_fixations_list = detectors.fixation_detection(
    #     simulated_scanpath[:, 1],
    #     simulated_scanpath[:, 0],
    #     simulated_scanpath[:, 2] * 1000.
    # )
    #
    # simulated_fixations_list = np.array(simulated_fixations_list)
    # if len(simulated_fixations_list) > 0:
    #     simulated_fixations_list = simulated_fixations_list[:, (3, 4, 0, 1)]

    simulated_fixations_list = []
    delta = 12
    i = delta
    while i < len(simulated_scanpath):
        simulated_fixations_list.append(
            [simulated_scanpath[i, 0], simulated_scanpath[i, 1], (i - delta) / 25., i / 25.]
        )
        i = i + delta
    simulated_fixations_list = np.array(simulated_fixations_list)

    return simulated_fixations_list


def save_fixations_list(
        SIMULATED_FIXATIONS_LISTS_FOLDER,
        model_name,
        dataset,
        dataset_subfolder,
        stimulus_name,
        simulated_scanpath_ID,
        fixations_list
):

    # define saving path (create folder in case they do not exist still)
    path = os.path.join(SIMULATED_FIXATIONS_LISTS_FOLDER)
    if not os.path.exists(path): os.mkdir(path)
    path = os.path.join(path, model_name)
    if not os.path.exists(path): os.mkdir(path)
    path = os.path.join(path, dataset)
    if not os.path.exists(path): os.mkdir(path)
    path = os.path.join(path, dataset_subfolder)
    if not os.path.exists(path): os.mkdir(path)
    path = os.path.join(path, stimulus_name)
    if not os.path.exists(path): os.mkdir(path)

    # save the array as file
    np.savetxt(
        os.path.join(
            path, str(simulated_scanpath_ID)
        ), fixations_list
    )


### 03


def save_fixmap(
        SIMULATED_FIXMAPS_FOLDER,
        path_to_fixations_list_folder,
        model_name,
        dataset,
        dataset_subfolder,
        stimulus_name,
        stimulus
):

    # Initialize Fixation map
    simulated_fixmap = np.zeros((np.shape(stimulus)[0], np.shape(stimulus)[1]))

    for simulated_fixations_list_file in os.listdir(
            path_to_fixations_list_folder
    ):

        simulated_fixations_list_file_fullpath = os.path.join(
            path_to_fixations_list_folder, simulated_fixations_list_file
        )

        if os.path.isfile(simulated_fixations_list_file_fullpath):
            simulated_fixations_list = read_simulated_scanpath(
                simulated_fixations_list_file_fullpath
            )
            for i in range(
                    len(simulated_fixations_list)
            ):
                simulated_fixmap[
                    int(simulated_fixations_list[i,0]),
                    int(simulated_fixations_list[i,1])
                ] = 1


    # define saving path (create folder in case they do not exist still)
    path = os.path.join(SIMULATED_FIXMAPS_FOLDER)
    if not os.path.exists(path): os.mkdir(path)
    path = os.path.join(path, model_name)
    if not os.path.exists(path): os.mkdir(path)
    path = os.path.join(path, dataset)
    if not os.path.exists(path): os.mkdir(path)
    path = os.path.join(path, dataset_subfolder)
    if not os.path.exists(path): os.mkdir(path)
    path = os.path.join(path, stimulus_name)
    if not os.path.exists(path): os.mkdir(path)

    # save fixmap as npz
    scipy.sparse.save_npz(
        os.path.join(
            path, "FixMap"
        ),
        scipy.sparse.csr_matrix(simulated_fixmap)
    )
