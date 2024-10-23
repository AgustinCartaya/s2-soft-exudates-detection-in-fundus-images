import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt    
import os
import random
# from tqdm import tqdm

IMG_DIMENSIONS_10 = (428, 284)


# -------- loading functions
def load_gt(img_path_name, img_size=None, filter_area=10):
    if os.path.isfile(img_path_name):
        if filter_area is None:
            return cv.imread(img_path_name, cv.IMREAD_UNCHANGED)
        gt = cv.imread(img_path_name, cv.IMREAD_UNCHANGED)
        real_sex, _  = cv.findContours(gt, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        real = []
        for contour in real_sex:
            if cv.contourArea(contour)>filter_area:
                real.append(contour)

        gt_real = np.zeros((gt.shape[0],gt.shape[1]), np.uint8)
        cv.drawContours(gt_real, real, -1, (255, 255, 255), -1)
        return gt_real
    
    else:
        # print("Image: ", img_path_name, "Not found")
        if img_size is not None:
            return np.zeros((img_size[0], img_size[1]), np.uint8)
        return None

# -------- Visualization functions
def notebook_show(img, description, separator=" ", show=False, explanation_depth=0, step_depth=0):
    if show and explanation_depth >= step_depth:
        str_description = ""

        if type(description) == str:
            str_description = description

        elif type(description) == tuple or type(description) == list :
            for i in range(len(description)):

                dec_piece = description[i]

                if type(description[i]) != str:
                    dec_piece = str(description[i])

                if i < len(description) -1:
                    str_description += dec_piece + separator
                else:
                    str_description += dec_piece

        print(str_description)
        imagen = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(imagen)
        plt.axis('off')
        plt.show()


def draw_contours(img, contours, thickness=-1, color=(255, 255, 255), contour_id=-1):
    if len(img.shape) < 3:
        img_with_contours = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    else:
        img_with_contours = img.copy()
    cv.drawContours(img_with_contours, contours, contour_id, color, thickness)
    return img_with_contours

def draw_contours_with_random_colors(contours, img_size):
    img_res = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    for contour_index in range(len(contours)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        cv.drawContours(img_res, contours, contour_index, (b,g,r), -1)
    return img_res

def draw_mask(img, mask, thickness=-1, mask_color=(0,255,0)):
    if mask is not None:
        contours,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return draw_contours(img, contours, thickness=thickness, color=mask_color, contour_id=-1)
    return img

def create_contours_mask(contours, img_dimensions, id=-1):
    mask = np.zeros((img_dimensions[0],img_dimensions[1]), np.uint8)
    cv.drawContours(mask, contours,  id, (255,255,255), -1)
    return mask

def draw_rois(img, rois, thickness=-1, color=(0, 0, 255)):
    if len(img.shape) < 3:
        img2 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        img2 = img.copy()

    for roi in rois:
        cv.rectangle(img2,
                     (roi[0], roi[1]),
                     (roi[2], roi[3]),
                     color, thickness)
    return img2


def get_contours_rois(contours, padding_th=None, img_shape=None):
    rois = []
    if padding_th is not None and img_shape is not None:
        for i in range(len(contours)):
            x, y, w, h = cv.boundingRect(contours[i])

            pad_x = 0
            pad_y = 0
            area = cv.contourArea(contours[i])

            for th, multiplier in padding_th:
                if area < th:
                    pad_x = w * multiplier
                    pad_y = h * multiplier
                    break
            rois.append((max(0, int(x-pad_x)), max(0, int(y-pad_y)), min(img_shape[1], int(x+w+pad_x)), min(img_shape[0], int(y+h+pad_y))))
    else:
        for i in range(len(contours)):
            x, y, w, h = cv.boundingRect(contours[i])
            rois.append((x, y, x+w, y+h))
    return rois


def filter_contour_area(contours, min_area, max_area=-1):
    candidates_filtered = []

    if max_area == -1:
        for contour in contours:
            area = cv.contourArea(contour)
            if area > min_area:
                candidates_filtered.append(contour) 
    else:
        for contour in contours:
            area = cv.contourArea(contour)
            if area > min_area and area < max_area:
                candidates_filtered.append(contour) 

    return candidates_filtered

def get_roi(img, roi):
    return img[roi[1]:roi[3], roi[0]:roi[2]]


def plot_points_2D(X, labels):
    """
    Plots points in 2D with colors according to their class.
    
    Parameters:
    X: np.array of shape (n_samples, 2) with the coordinates of the points.
    y: list or np.array of size (n_samples) with the classes (0 or 1) of each point.
    """
    # Convert lists to NumPy arrays for better data handling
    X = np.array(X)
    labels = np.array(labels)
    
    # Create masks for class 0 and class 1
    mask_0 = (labels == 0)
    mask_1 = (labels == 1)

    # Plot the points
    plt.scatter(X[mask_0, 0], X[mask_0, 1], color='blue', label='No SE')
    plt.scatter(X[mask_1, 0], X[mask_1, 1], color='red', label='SE')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# -------- image redimentioning functions
def get_resize_dimensions(img, scale_percent=1):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    return (width, height)

def resize(img, scale):
    return cv.resize(img, get_resize_dimensions(img, scale))

def get_image_10_percent(img):
    sf =  round(IMG_DIMENSIONS_10[0]/img.shape[1], 2)
    # sf =  0.1
    return (resize(img, sf), sf)

def resize_contours(contours, sf):
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            contours[i][j][0][0] = sf * contours[i][j][0][0]
            contours[i][j][0][1] = sf * contours[i][j][0][1]

def resize_rectangle(rect, sf):
    return (int(rect[0]*sf), int(rect[1]*sf), int(rect[2]*sf), int(rect[3]*sf))

def resize_rectangles(rects, sf):
    rects_resized = []
    for rect in rects:
        rects_resized.append(resize_rectangle(rect, sf))
    return rects_resized

# -------- geometric functions
def resize_circle(circle, sf, translate=False):
    if translate:
        return ((int(circle[0][0]*sf), int(circle[0][1]*sf)),  int(circle[1]*sf))
    else:
        return ((circle[0][0], circle[0][1]),  int(circle[1]*sf))
    

# -------- masks functions
def apply_mask(img, mask, mask_value=0):
    img_masked = img.copy()
    if mask_value == -1:
        img_masked[mask==0] = 0
    else:
        img_masked[mask==255] = mask_value
    return img_masked


def create_circle_mask(circle, img_dimensions):
    mask = np.zeros((img_dimensions[0],img_dimensions[1]), np.uint8)
    cv.circle(mask, circle[0], circle[1], (255,255,255), -1)
    return mask


def add_patch(img, patch, pos, circular=False):
    if circular:
        pos = [pos[0][0]-pos[1], pos[0][1]-pos[1], 2*pos[1], 2*pos[1]]
    img_restored = img.copy()
    img_restored[pos[1]:pos[1]+pos[3], pos[0]:pos[0]+pos[2]] = patch
    return img_restored   


# -------- histogram eq functions
def gamma_correction(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv.LUT(src, table)


# -------- Binary morphology
def break_big_elements(img, size_th=400, se=5):
    candidates, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    small_elements = []

    for i in range(len(candidates)):
        area = cv.contourArea(candidates[i])
        if area < size_th:
            small_elements.append(candidates[i])

    SE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (se,se))
    img_open = cv.morphologyEx(img, cv.MORPH_OPEN, SE)

    cv.drawContours(img_open, small_elements, -1, (255, 255, 255), -1)
    return img_open


# -------- metrics
# def label_contours(candidates_contours, gt_contours, img_shape, overlap_threshold=0.5):
#     gt_len = len(gt_contours)
#     candidates_len = len(candidates_contours)

#     if gt_len == 0:
#         return np.zeros(candidates_len), 0

#     candidates_match = np.full((gt_len, 2), -1)
#     candidates_labels = np.zeros(candidates_len)
#     TP = 0

#     for i in tqdm(range(candidates_len)):
#         mask_one = np.zeros((img_shape[0],img_shape[1]), np.uint8)
#         cv.drawContours(mask_one, candidates_contours, i, (255, 255, 255), -1)

#         for j in range(gt_len):
#             gt_one = np.zeros((img_shape[0],img_shape[1]), np.uint8)
#             cv.drawContours(gt_one, gt_contours, j, (255, 255, 255), -1)  

#             intersection = cv.bitwise_and(mask_one, gt_one)
#             area_intersection = cv.countNonZero(intersection)
#             union = cv.bitwise_or(mask_one, gt_one)
#             area_union = cv.countNonZero(union)

#             if area_intersection > 0: 
#                 overlap = area_intersection / area_union
#                 if overlap > overlap_threshold:
#                     if candidates_match[j, 0] != -1:
#                         if overlap>candidates_match[j, 1]:
#                             candidates_labels[candidates_match[j, 0]] = 0
#                             candidates_labels[i] = 1
#                             candidates_match[j, 0] = i
#                             candidates_match[j, 1] = overlap
#                             break

#                     else:
#                         candidates_match[j, 0] = i
#                         candidates_match[j, 1] = overlap
#                         candidates_labels[i] = 1
#                         TP += 1
#                         break

#     FN = len(gt_contours) - TP

#     if (TP + FN)   == 0:
#         img_sensitivity = 0
#     else:
#         img_sensitivity = TP / (TP + FN)  

#     return candidates_labels, img_sensitivity