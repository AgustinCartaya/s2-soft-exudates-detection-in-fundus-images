from .functions import *
from .basic_sex_detection import BasicSExDetection
from .advanced_od_removal import AdvancedODRemoval

class AdvancedSExDetection:

    COMPLETP_CLAHE_CLIP = 14
    COMPLETP_MEDIAN_BLUR_SIZE = 3
    COMPLETP_GAMMA = 0.5

    def __init__(self, notebook=False, explanation_depth=0):
        self.notebook = notebook
        self.explanation_depth = explanation_depth


    def exudates_detection(self, img):
        # basic detection
        basic_sex_candidates, basic_rois  = BasicSExDetection().exudates_detection(img)
        
        # od removal
        img_no_od = AdvancedODRemoval().remove_od(img)
        # notebook_show(img_no_od, "Step 1: Remove OD", show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)


        # resize image and rois
        sf = 0.4
        img_no_od = resize(img_no_od, sf)
        basic_rois_resized = resize_rectangles(basic_rois, sf)
        # preprocess image
        img_no_od_pre = self.__complet_preprocessing(img_no_od)

        # #-------------- added only to show the steps
        # if self.notebook and self.explanation_depth >=1:
        #     self.__complet_preprocessing_notebook(img_no_od)
        #     img_notebook_rois = draw_rois(img_no_od_pre, basic_rois_resized, thickness=2, color=(0, 255, 255))
        #     notebook_show(img_notebook_rois, "Step 6: Work with each roi independently", show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)
        
        #     sub_preprocessing_step_names = [["Preprocessing the whole image depending on the ROI mean"],
        #                                     ["Median blur with windows size of =",15, "x", 15],
        #                                     ["Gamma correction with sigma =", 1/0.24],
        #                                     ["Otsu threshold",] ]
        #     for sub_preprocessing_step in [0, 1, 2, 3]:
        #         img_rois_advanced_preproced_notebook = []
        #         for roi in basic_rois_resized:
        #             img_roi = img_no_od_pre[roi[1]:roi[3], roi[0]:roi[2]]

        #             mean_intensity = img_roi.mean()
        #             if mean_intensity <= 34:
        #                 clahe_gamma = 66
        #             elif mean_intensity <= 130:
        #                 clahe_gamma= -0.4*mean_intensity + 90
        #             else:
        #                 clahe_gamma = 35

        #             img_no_od_pre = self.__complet_preprocessing(img_no_od, clahe_clip=clahe_gamma, gamma=clahe_gamma/100)
        #             img_roi = img_no_od_pre[roi[1]:roi[3], roi[0]:roi[2]]
        #             if sub_preprocessing_step == 0:
        #                 img_roi_processed = img_roi
        #             else:
        #                 img_roi_processed =  self.__process_roi_notebook(img_roi, sub_preprocessing_step-1)
        #             img_rois_advanced_preproced_notebook.append(img_roi_processed)

        #         # reconstruction original img
        #         img_notebook_reconstructed = self.__reconstruct(basic_rois_resized, img_rois_advanced_preproced_notebook, (int(img.shape[0]*sf), int(img.shape[1]*sf)))
        #         img_notebook_rois = draw_rois(img_notebook_reconstructed, basic_rois_resized, thickness=1, color=(0, 255, 255))
        #         notebook_show(img_notebook_rois, list(("Step", sub_preprocessing_step + 7, ":")) + sub_preprocessing_step_names[sub_preprocessing_step], show=self.notebook, explanation_depth=self.explanation_depth, step_depth=2)
        # #-------------- end added only to show the steps


        img_rois_advanced_preproced = []
        # study rois
        for roi in basic_rois_resized:
            img_roi = img_no_od_pre[roi[1]:roi[3], roi[0]:roi[2]]

            # intensity modification
            mean_intensity = img_roi.mean()
            if mean_intensity <= 34:
                clahe_gamma = 66
            elif mean_intensity <= 130:
                clahe_gamma= -0.4*mean_intensity + 90
            else:
                clahe_gamma = 35

            img_no_od_pre = self.__complet_preprocessing(img_no_od, clahe_clip=clahe_gamma, gamma=clahe_gamma/100)
            img_roi = img_no_od_pre[roi[1]:roi[3], roi[0]:roi[2]]
            img_roi_processed =  self.__process_roi(img_roi)
            img_rois_advanced_preproced.append(img_roi_processed)

        # reconstruction original img
        img_reconstructed = self.__reconstruct(basic_rois_resized, img_rois_advanced_preproced, (int(img.shape[0]*sf), int(img.shape[1]*sf)))
        
        # fast dilate
        SE_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        img_reconstructed = cv.morphologyEx(img_reconstructed, cv.MORPH_DILATE, SE_dilate, iterations=1)
        # notebook_show(img_reconstructed, ("Step 11: Global dilate with elliptical element of size", 7, "x", 7), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        # find contours
        advance_sex_candidates, _ = cv.findContours(img_reconstructed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # dilate by contours
        advance_sex_candidates = self.__dilate_contours(advance_sex_candidates, img.shape, dilate=3) # Total sensibility: 0.73
        
        # #-------------- added for the notebook
        # if self.notebook and self.explanation_depth >= 1: 
        #     initial_contours_number = len(advance_sex_candidates)
        #     img_notebook_dilate_contours = draw_contours_with_random_colors(advance_sex_candidates, img_no_od_pre.shape)
        #     notebook_show(img_notebook_dilate_contours, ("Step 12: Individual dilate with elliptical element of size", 3, "x", 3), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)
        # #-------------- end added for the notebook

        # resize to the original size
        resize_contours(advance_sex_candidates, 1/sf)
        advance_sex_candidates = filter_contour_area(advance_sex_candidates, min_area=773*0.25, max_area=54468*1.5)
        
        # #-------------- added for the notebook
        # if self.notebook and self.explanation_depth >= 1: 
        #     img_notebook_filtered_elements = draw_contours_with_random_colors(advance_sex_candidates, img.shape)
        #     notebook_show(img_notebook_filtered_elements, ("Step 13: Filter contours by area >", 773*0.25, "and area <", 54468*1.5, "Removed contours =", initial_contours_number - len(advance_sex_candidates)), 
        #                 show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)
        # #-------------- end added for the notebook

        # obtain rois
        advance_rois = get_contours_rois(advance_sex_candidates)
        return advance_sex_candidates, advance_rois




    def __process_roi(self, img_roi):
        img_out = cv.medianBlur(img_roi, 15) #57
        img_out = gamma_correction(img_out, 24/100) #24
        th, img_out = cv.threshold(img_out, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return img_out


    def __complet_preprocessing(self, img, clahe_clip=COMPLETP_CLAHE_CLIP, gamma=COMPLETP_GAMMA, median_blur=COMPLETP_MEDIAN_BLUR_SIZE):
        green = cv.extractChannel(img, 1) 
        clahe = cv.createCLAHE(clipLimit=clahe_clip) #14
        img_clahe = clahe.apply(green)
        img_median_blur = cv.medianBlur(img_clahe, median_blur) # 3
        img_gamma_correction = gamma_correction(img_median_blur, gamma) # 0.5
        return img_gamma_correction



    def __reconstruct(self, rois_coord, rois_imgs, img_shape):
        out = np.zeros((img_shape[0], img_shape[1]), np.uint8)
        for i in range(len(rois_imgs)):
            out_roi = out[rois_coord[i][1]:rois_coord[i][3], rois_coord[i][0]:rois_coord[i][2]]
            resultado = cv.bitwise_or(out_roi, rois_imgs[i])
            out[rois_coord[i][1]:rois_coord[i][3], rois_coord[i][0]:rois_coord[i][2]] = resultado

        return  out


    def __dilate_contours(self, advance_sex_candidates, img_shape, dilate=5):
        contour_len = len(advance_sex_candidates)
        for i in range(contour_len):
            mask_one = np.zeros((img_shape[0],img_shape[1]), np.uint8)
            cv.drawContours(mask_one, advance_sex_candidates, i, (255, 255, 255), -1)

            SE_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate, dilate))
            mask_one = cv.morphologyEx(mask_one, cv.MORPH_DILATE, SE_dilate, iterations=1)
            
            contours, _ = cv.findContours(mask_one, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            advance_sex_candidates = list(advance_sex_candidates)
            advance_sex_candidates[i] = contours[0]
            advance_sex_candidates = tuple(advance_sex_candidates)

        return advance_sex_candidates

    
    #-------------- added only to show the steps
    def __process_roi_notebook(self, img_roi, sub_preprocessing_step=0):
        img_out = cv.medianBlur(img_roi, 15) #57
        if sub_preprocessing_step > 0:
            img_out = gamma_correction(img_out, 24/100) #24
        if sub_preprocessing_step > 1:
            th, img_out = cv.threshold(img_out, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return img_out

    def __complet_preprocessing_notebook(self, img, clahe_clip = COMPLETP_CLAHE_CLIP, gamma = COMPLETP_GAMMA, median_blur = COMPLETP_MEDIAN_BLUR_SIZE, notebook=False, explanation_depth=0):
        green = cv.extractChannel(img, 1) 
        # notebook_show(green, "Step 2: extract green channel", show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        clahe = cv.createCLAHE(clipLimit=clahe_clip) #14
        img_clahe = clahe.apply(green)
        # notebook_show(img_clahe, ("Step 3: CLAHE with clip =", clahe_clip), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        img_median_blur = cv.medianBlur(img_clahe, median_blur) # 3
        # notebook_show(img_median_blur, ("Step 4: Median blur with  kernel size =", median_blur), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        img_gamma_correction = gamma_correction(img_median_blur, gamma) # 0.5
        # notebook_show(img_gamma_correction, ("Step 5: Gamma correction with sigma =", 1/gamma), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        return img_gamma_correction
    #-------------- end added only to show the steps
