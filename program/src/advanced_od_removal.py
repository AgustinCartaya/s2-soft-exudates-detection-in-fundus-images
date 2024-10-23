
from .functions import *
from .basic_od_removal import BasicODRemoval

import math 


class AdvancedODRemoval:
    GAMMA_SIGMA = 20 
    MIN_CIRCULARITY = 82

    HOUGH_DP=4
    HOUGH_MINRADIUS=160
    HOUGH_MAXRADIUS=320

    def __init__(self, notebook=False, explanation_depth=0):
        self.notebook = notebook
        self.explanation_depth = explanation_depth

        self.gamma_intensities = []
        self.th_intensities = []


    def remove_od(self, img):
        
        roi = BasicODRemoval().get_od_roi(img)

        # obtain working area
        img_working_area = self.__get_croped_circle_roi(img, roi)
        # notebook_show(img_working_area, "Step 1: Obtain ROI from basic od removal", show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        # apply circular mask to the wirking are to focus the the OD
        img_working_area_masked = apply_mask(img_working_area, create_circle_mask(((roi[1],roi[1]), roi[1]), (2*roi[1] , 2*roi[1] )), -1)
        # notebook_show(img_working_area_masked, "Step 2: Apply circular mask to the ROI", show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        # advance od segmentation
        pre_segmented_od_mask = self.__pre_segmentation(img_working_area_masked)
        segmented_od_mask = self.__circularity_segmentation(pre_segmented_od_mask)

        segmented_od_bw = cv.bitwise_not(segmented_od_mask)
        img_working_area_no_od = apply_mask(img_working_area,segmented_od_bw,-1)

        img_segmented_od = add_patch(img, img_working_area_no_od, roi, circular=True)

        return img_segmented_od


    def get_od_mask(self, img):
        roi = BasicODRemoval().get_od_roi(img)

        img_working_area = self.__get_croped_circle_roi(img, roi)
        img_working_area_masked = apply_mask(img_working_area, create_circle_mask(((roi[1],roi[1]), roi[1]), (2*roi[1] , 2*roi[1] )), -1)

        # advance od segmentation
        pre_segmented_od_mask = self.__pre_segmentation(img_working_area_masked)
        segmented_od_mask = self.__circularity_segmentation(pre_segmented_od_mask)

        od_mask = np.zeros((img.shape[0],img.shape[1]), np.uint8)
        od_mask = add_patch(od_mask, segmented_od_mask, roi, circular=True)

        return od_mask
    

    def __get_croped_circle_roi(self, img, roi, masked=False):
        # obtain cropping area
        x, y, w_h = roi[0][0]-roi[1], roi[0][1]-roi[1], 2*roi[1] 
        img_croped_roi = img[y:y+w_h, x:x+w_h]

        # apply circular mask to cropped area
        if masked:
            img_masked = apply_mask(img_croped_roi, create_circle_mask(((roi[1],roi[1]), roi[1]), (w_h, w_h)), -1)
            return img_masked
        else:
            return img_croped_roi
        

    def __pre_segmentation(self, roi_img):
        img_v = cv.extractChannel(cv.cvtColor(roi_img, cv.COLOR_RGB2HSV), 2)
        # notebook_show(img_v, "Step 3: The space color of the image is changed to HSV and the Value chanel (V) is extracted", show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        mean_intensity = cv.mean(img_v)[0]
        self.gamma_intensities.append(mean_intensity)

        if mean_intensity < 150:
            gamma_sigma = 0.3
        elif mean_intensity > 190:
            gamma_sigma = 0.01
        else:
            gamma_sigma = -0.000061*mean_intensity**2 + 0.0167*mean_intensity - 0.8678

        img_v_gamma = gamma_correction(img_v, gamma_sigma)
        # notebook_show(img_v_gamma, ("Step 4: Gamma correction with sigma =", 1/gamma_sigma), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        kernel_size = 51
        img_v_gamma_blur = cv.medianBlur(img_v_gamma, kernel_size)
        # notebook_show(img_v_gamma_blur, ("Step 5: Median blur with  kernel size =", kernel_size), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        otsu_th, img_th = cv.threshold(img_v_gamma_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        # notebook_show(img_th, ("Step 6: Otsu threshold with th =", otsu_th), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        se_size = 55
        SE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (se_size,se_size))

        img_open = cv.morphologyEx(img_th, cv.MORPH_OPEN, SE, iterations=2)
        # notebook_show(img_open, ("Step 7: Binary open with elliptical SE size =", se_size, "x", se_size), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        img_close = cv.morphologyEx(img_open, cv.MORPH_CLOSE, SE, iterations=2)
        # notebook_show(img_close, ("Step 8: Binary close with elliptical SE size =", se_size, "x", se_size), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        return img_close

        
    def __circularity_segmentation(self, pre_img):
        # find contours
        candidates, _ = cv.findContours(pre_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        if len(candidates) == 0:
            return None
        
        candidate_features = []

        # filter elements by area
        candidate_counter = 0
        for contour in candidates:
            area = cv.contourArea(contour)
            p = cv.arcLength(contour, True)
            C = 4 * math.pi * area / (p*p)
        
            candidate_features.append((candidate_counter, area, C, p))
            candidate_counter +=1

        # obtaining biggest candidate
        candidate_features.sort(key=lambda x: x[1])
        biggest_candidate = candidate_features[-1]

        # draw biggest candidate
        img_bigger_candidate = create_contours_mask(candidates, pre_img.shape, id=biggest_candidate[0])
        # notebook_show(img_bigger_candidate, "Step 9: OD bigest candidate", show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        if biggest_candidate[2] < self.MIN_CIRCULARITY/100:
            img_hough_candidate = self.__find_best_hough_circle(img_bigger_candidate)
            # notebook_show(img_hough_candidate, ("Step 10: OD hough candidate because circularity =", biggest_candidate[2], "<",  self.MIN_CIRCULARITY/100), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)
            return img_hough_candidate
        else:
            return img_bigger_candidate


    def __find_best_hough_circle(self, pre_img):
        
        SE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (60,60))
        pre_img = cv.morphologyEx(pre_img, cv.MORPH_OPEN, SE, iterations=2)

        circles = cv.HoughCircles(pre_img, cv.HOUGH_GRADIENT, dp=self.HOUGH_DP, minDist=10000, param1=255, param2=30,
                                minRadius=self.HOUGH_MINRADIUS, maxRadius=self.HOUGH_MAXRADIUS)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            circle = ((circles[0][0][0], circles[0][0][1]), circles[0][0][2])
            return create_circle_mask(circle,(pre_img.shape[0], pre_img.shape[1]))        
        else:
            print("ADVANCED OD REMOVAL ERROR FINDING BEST HOUGH CIRCLE IN IMAGE")
            return pre_img
