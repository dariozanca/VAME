import numpy as np
import cv2
import os

model = r'\D1'  # or r'\DW2'
category = r'\Social'
image_number = r'\029'


image_path_base = r'STIMULI\\CAT2000_TRAIN\\'
result_path_base = r'SIMULATED_FIXATIONS_LISTS'

image_path = image_path_base + category + image_number + '.jpg'
results_path = result_path_base + model + r'\CAT2000_TRAIN' + category + image_number


# read the image
image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the file
h, w = image.shape[0:2]

TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
TEXT_SCALE = 1.5
TEXT_THICKNESS = 2

visualization_mode = "numbers"
# visualization_mode = "lines"

i = 0
for filename in os.listdir(results_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Reload the image the file
    specific_result_path = os.path.join(results_path, filename)
    a = np.loadtxt(specific_result_path, usecols=(0, 1), dtype=np.float32)
    for j in range(a.shape[0]):

        center = (a[j,1],a[j,0]) # (horizzontal_position, vertical_position)

        if visualization_mode == "numbers":
            cv2.circle(image,center=center, radius=30, color=(0,j*25,200),thickness= 5)
            text = "%i"%j
            text_size, _ = cv2.getTextSize(text, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
            text_origin = (int(center[0] - text_size[0] / 2), int(center[1] + text_size[1] / 2))
            cv2.putText(image, text, text_origin, TEXT_FACE, TEXT_SCALE, (0,j*25,200), TEXT_THICKNESS, cv2.LINE_AA)
        elif visualization_mode == "lines":
            cv2.circle(image, center, 10, (255, 0, 0), -1)
            if j > 0:
                cv2.line(image, previous_center, center, (255, 0, 0), 3, cv2.LINE_AA)
            previous_center = center

    window_name = "Fixations List %i - model " %i + model
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a window for display.
    cv2.resizeWindow(window_name, w // 2, h // 2)
    cv2.moveWindow(window_name, 300, 100)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    i+=1





