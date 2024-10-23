
import math

from .functions import *


class BasicODRemoval:
    CLAHE_TH_LOW= 116 
    CLAHE_TH_HIGH= 135 

    TH=205
    SE_SIZE=10
    HOUGH_DP=4
    HOUGH_MINRADIUS=15
    HOUGH_MAXRADIUS=40
    CLAHE_CLIP_LOW=1
    CLAHE_CLIP_HIGH=6

    def __init__(self, notebook=False, explanation_depth=0):
        self.notebook = notebook
        self.explanation_depth = explanation_depth


    def remove_od(self, img, circle_sf=1):
        # obtain od mask
        od_mask = self.get_od_mask(img, circle_sf)

        # apply mask
        img_no_od = apply_mask(img, od_mask)
        return img_no_od
    

    def get_od_roi(self, img):
        # resize image to the 10% of the full scale fundus image (4288, 2848)
        img_resized, sf = get_image_10_percent(img)

        # obtain od circle of the 10% image
        roi_od_circle = self.__get_OD_circle(img_resized)

        # convert obtained od circle into roi
        if roi_od_circle[1] < self.HOUGH_MINRADIUS * 1.5:
            roi_od_circle = resize_circle(roi_od_circle, 2, translate=False) 
        elif roi_od_circle[1] > self.HOUGH_MAXRADIUS * 0.8:
            roi_od_circle = resize_circle(roi_od_circle, 1.2, translate=False) 
        else:
            roi_od_circle = resize_circle(roi_od_circle, 1.5, translate=False) 

        # scale roi for the img size 
        roi_od_circle_reziced = resize_circle(roi_od_circle, 1/sf, translate=True)

        return roi_od_circle_reziced
    

    def get_od_mask(self, img, circle_sf=1):
        
        # resize image to the 10% of the full scale fundus image (4288, 2848)
        img_resized, sf = get_image_10_percent(img)

        # obtain od circle of the 10% image
        od_circle = self.__get_OD_circle(img_resized)
        
        # if self.notebook and self.explanation_depth > 1:
        #     img_notebook = img_resized.copy()
        #     cv.circle(img_notebook, od_circle[0], od_circle[1], (255, 255, 255), 2)
        #     notebook_show(img_notebook, "Step 5: Choosen OD circle", show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)


        # scale od circle for the img size
        od_circle_reziced = resize_circle(od_circle, 1/sf, translate=True)

        # resize circle if necessary (used in basic soft exudates detection)
        od_circle_reziced = resize_circle(od_circle_reziced, circle_sf, translate=False)

        # obtain od mask
        od_mask = create_circle_mask(od_circle_reziced, (img.shape[0], img.shape[1]))
        return od_mask
    


    def __get_OD_circle(self, img_resized):
        # --------- preprocessing
        img_chanel_v = cv.extractChannel(cv.cvtColor(img_resized, cv.COLOR_RGB2HSV), 2)
        # notebook_show(img_chanel_v, "Step 1: The space color of the image is changed to HSV and the Value chanel (V) is extracted", show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        mean_intensity = cv.mean(img_chanel_v)

        # histogram equalization
        clahe_clip = 4
        only_hough = False
        if int(mean_intensity[0]) < self.CLAHE_TH_LOW:
            clahe_clip = self.CLAHE_CLIP_LOW
        elif int(mean_intensity[0]) > self.CLAHE_TH_HIGH:
            clahe_clip = self.CLAHE_CLIP_HIGH
            only_hough = True
        

        # histogram equalization
        clahe = cv.createCLAHE(clipLimit=clahe_clip)
        img_pre_clahe = clahe.apply(img_chanel_v)
        # notebook_show(img_pre_clahe, ("Step 2: CLAHE with clip =", clahe_clip), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        # binarization
        _,img_th = cv.threshold(img_pre_clahe, self.TH, 255, cv.THRESH_BINARY)
        # notebook_show(img_th, ("Step 3: Binary threshold with th =", self.TH), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=2)

        # removing small white noise
        SE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.SE_SIZE, self.SE_SIZE))
        img_th = cv.morphologyEx(img_th, cv.MORPH_OPEN, SE)
        # notebook_show(img_th, ("Step 4: Binary open with elliptical SE size =", self.SE_SIZE, "x", self.SE_SIZE), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        # obtaining OD candidates
        contour_candidate = self.__disk_segmentation_candidate_contours(img_th)
        hough_candidate = self.__disk_segmentation_candidate_hough(img_th)

        # creating OD mask
        if contour_candidate is None or hough_candidate is None:
            print("ERROR CREATING OD MASK FOR IMG ")
            return None
        else:
            if self.notebook and self.explanation_depth >= 1:
                img_notebook = cv.cvtColor(img_th, cv.COLOR_GRAY2RGB)
                cv.circle(img_notebook, contour_candidate[0], contour_candidate[1], (255, 255,0), 2)
                cv.circle(img_notebook, hough_candidate[0], hough_candidate[1], (0, 255, 255), 2)
                notebook_show(img_notebook, "OD candidates, hough candidate = Yellow; contour candidate = Cyan", show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

            candidates_distance = (contour_candidate[0][0] - hough_candidate[0][0])**2 + (contour_candidate[0][1] - hough_candidate[0][1])**2
            if only_hough or candidates_distance > hough_candidate[1]**2:
                return hough_candidate
            else:
                mean_center = ( int((contour_candidate[0][0] + hough_candidate[0][0])/2), int((contour_candidate[0][1] + hough_candidate[0][1])/2))
                mean_radius = int((contour_candidate[1] + hough_candidate[1]) / 2)
                return (mean_center, mean_radius)


    def __disk_segmentation_candidate_contours(self, img):
        # find contours
        candidates, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
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

        # obtaining best OD candidate
        candidate_features.sort(key=lambda x: x[1])
        best_candidate = candidate_features[-1]

        (x,y), radius = cv.minEnclosingCircle(candidates[best_candidate[0]])
        circle = ((int(x),int(y)), int(radius))
        return circle


    def __disk_segmentation_candidate_hough(self, img):
        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=self.HOUGH_DP, minDist=10000, param1=255, param2=30,
                                minRadius=self.HOUGH_MINRADIUS, maxRadius=self.HOUGH_MAXRADIUS)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            return ((circles[0][0][0], circles[0][0][1]), circles[0][0][2])
        else:
            print("PROBLEM WITH THE IMAGE")
            return None
