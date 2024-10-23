

import pathlib
import os



def check_image(img):
    # check if the image is a valid image
    message = ''
    if img is None:
        message = 'The image is None'
        return message, False
    if img.shape[0] == 0 or img.shape[1] == 0:
        message = 'The image has a dimension of 0'
        return message, False
    if len(img.shape) != 3:
        message = 'The image is not a color image'
        return message, False
    if img.shape[2] != 3:
        message = 'The image does not have 3 channels'
        return message, False
    if img.shape[1] < 3000 or img.shape[0] < 1500:
        message = 'The image is too small, minimum size is 3000x1500 and the image is {}x{}'.format(img.shape[1], img.shape[0])
        return message, False
    return "OK", True
