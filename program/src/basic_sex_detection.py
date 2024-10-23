from .functions import *
from .basic_od_removal import BasicODRemoval


class BasicSExDetection:

    CLAHE_CLIP = 14 
    BINARY_TH = 100
    BINARY_NEIGHBORS = 53
    MEAN_DIFFERENCE = 2
    OPEN_FACTOR = 3
    ROI_SEX_DILATE = 3

    GAUSSIAN_W=5
    SUB_P_GAMMA_C=0.3
    SUBD_WINDOW_SIZE = 35
    MIN_SIZE = 5

    def __init__(self, notebook=False, explanation_depth=0):
        self.notebook = notebook
        self.explanation_depth = explanation_depth

    def exudates_detection(self, img):
        
        od_enlargement_factor = 1.5
        img_no_od = BasicODRemoval().remove_od(img, circle_sf=od_enlargement_factor)
        # notebook_show(img_no_od, ("Step 1: Remove OD region enlarged by a factor of", od_enlargement_factor), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        img_resized, sf = get_image_10_percent(img_no_od)
        # notebook_show(img_resized, ("Step 1.2: Reduce the image by a scale factor of ", sf), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=3)


        img_preprocesed = self.__preprocessing(img_resized)
        
        img_reconstructed = self.__reconstruction(self.__subdivision(img_preprocesed))
        
        img_filtered_by_shape = self.__filter_shape(img_reconstructed)

        candidates, _ = cv.findContours(img_filtered_by_shape, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        # filtering small elements
        candidates = filter_contour_area(candidates, min_area=3)
        # if self.notebook and self.explanation_depth >= 1: 
        #     img_notebook_1 = create_contours_mask(candidates, img_resized.shape)
        #     notebook_show(img_notebook_1, ("Step 14: Filter small elements with size <", 3), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        # obtain bounding boxes
        rois = get_contours_rois(candidates, 
                        [(30, 0.7), (50, 0.6), (100, 0.4), (150, 0.3), (300, 0.2), (400, 0.1)],
                        img.shape)
        
        # resize contours to the original image
        resize_contours(candidates, 1/sf)
        rois = resize_rectangles(rois, 1/sf)

        return (candidates, rois)
    

    
    def __preprocessing(self, img):
        green = cv.extractChannel(img, 1)
        # notebook_show(green, "Step 2: extract green channel", show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        clahe = cv.createCLAHE(clipLimit=self.CLAHE_CLIP)
        img_clahe = clahe.apply(green)
        # notebook_show(img_clahe, ("Step 3: CLAHE with clip =", self.CLAHE_CLIP), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        kernel_size = 3
        img_median_blur = cv.medianBlur(img_clahe, kernel_size)
        # notebook_show(img_clahe, ("Step 4: Median blur with  kernel size =", kernel_size), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        gamma_sigma = 0.5
        img_gamma_correction = gamma_correction(img_median_blur, gamma_sigma)
        # notebook_show(img_gamma_correction, ("Step 5: Gamma correction with sigma =", 1/gamma_sigma), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)

        return img_gamma_correction
    

    def __reconstruction(self, args):
        pieces, original_dimensions, self.SUBD_WINDOW_SIZE = args 
        img_rec = np.zeros((original_dimensions[0], original_dimensions[1]), np.uint8)
        for piece, patch_init in pieces:
            patch_size = (min(self.SUBD_WINDOW_SIZE, original_dimensions[0]-patch_init[0]),
                        min(self.SUBD_WINDOW_SIZE, original_dimensions[1]-patch_init[1]))
            img_rec[patch_init[0]: patch_init[0]+patch_size[0], patch_init[1]: patch_init[1]+patch_size[1]] = piece[:patch_size[0], :patch_size[1]]
        return img_rec


    def __subdivision(self, img):
        padding = (self.SUBD_WINDOW_SIZE-img.shape[0] % self.SUBD_WINDOW_SIZE, self.SUBD_WINDOW_SIZE-img.shape[1] % self.SUBD_WINDOW_SIZE)
        img_padding = cv.copyMakeBorder(img, 0, padding[0], 0, padding[1], cv.BORDER_CONSTANT, value=(0,0, 0))

        # #-------------- added for the notebook
        # if self.notebook and self.explanation_depth >=1:
        #     notebook_windows = []
        #     for y in range(0, img_padding.shape[0], self.SUBD_WINDOW_SIZE):
        #         for x in range(0, img_padding.shape[1], self.SUBD_WINDOW_SIZE):
        #             notebook_windows.append((x, y, x+self.SUBD_WINDOW_SIZE, y+self.SUBD_WINDOW_SIZE))
            
        #     img_notebook_windows = draw_rois(img_padding, notebook_windows, thickness=1, color=(0,255,255))
        #     notebook_show(img_notebook_windows, ("Step 6: add padding =", padding, "to the image for then subdivide in tiles of =",self.SUBD_WINDOW_SIZE, "x", self.SUBD_WINDOW_SIZE), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=1)
        
        #     sub_preprocessing_step_names = [["Gaussain blur with windows size of = ",self.GAUSSIAN_W, "x", self.GAUSSIAN_W],
        #                                     ["Gamma correction with sigma =", 1/self.SUB_P_GAMMA_C],
        #                                     ["Otsu threshold",] ]
        #     for sub_preprocessing_step in [0, 1, 2]:
        #         regions_notebook = []
        #         for y in range(0, img_padding.shape[0], self.SUBD_WINDOW_SIZE):
        #             for x in range(0, img_padding.shape[1], self.SUBD_WINDOW_SIZE):
        #                 crop_img = img_padding[y:y+self.SUBD_WINDOW_SIZE, x:x+self.SUBD_WINDOW_SIZE]
        #                 if crop_img.mean() > 5:
        #                     crop_img_processed = self.__sub_preprocessing_notebook(crop_img, sub_preprocessing_step=sub_preprocessing_step)
        #                     regions_notebook.append((crop_img_processed, (y, x)))
        #         img_notebook_reconstructed = self.__reconstruction((regions_notebook, img.shape, self.SUBD_WINDOW_SIZE))
                
        #         img_notebook_reconstructed = cv.copyMakeBorder(img_notebook_reconstructed, 0, padding[0], 0, padding[1], cv.BORDER_CONSTANT, value=(0,0, 0))
        #         img_notebook_reconstructed = draw_rois(img_notebook_reconstructed, notebook_windows, thickness=1, color=(0,255,255))
        #         notebook_show(img_notebook_reconstructed, list(("Step", sub_preprocessing_step + 7, ":")) + sub_preprocessing_step_names[sub_preprocessing_step], show=self.notebook, explanation_depth=self.explanation_depth, step_depth=2)
        # #-------------- end added for the notebook
            
        regions = []
        for y in range(0, img_padding.shape[0], self.SUBD_WINDOW_SIZE):
            for x in range(0, img_padding.shape[1], self.SUBD_WINDOW_SIZE):
                crop_img = img_padding[y:y+self.SUBD_WINDOW_SIZE, x:x+self.SUBD_WINDOW_SIZE]

                if crop_img.mean() > 5:
                    crop_img_processed = self.__sub_preprocessing(crop_img)
                    regions.append((crop_img_processed, (y, x)))
        return (regions, img.shape, self.SUBD_WINDOW_SIZE)
    

    def __sub_preprocessing(self, img):
        gaussian_w = self.GAUSSIAN_W
        if self.GAUSSIAN_W%2==0:
            gaussian_w+=1
        img_gaussian = cv.GaussianBlur(img, (gaussian_w, gaussian_w), 0)

        img_gamma = gamma_correction(img_gaussian, self.SUB_P_GAMMA_C)
        _, img_th = cv.threshold(img_gamma, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        return img_th
    
    def __filter_shape(self, img):
        # breaking big elements
        img_big_elements_beaked = break_big_elements(img, size_th=300, se=5)
        # notebook_show(img_big_elements_beaked, ("Step 10: Sequential filter for breaking big elements of size >", 300, "with Opening se =", 5), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=3)

        img_big_elements_beaked = break_big_elements(img_big_elements_beaked, size_th=300, se=7)
        # notebook_show(img_big_elements_beaked, ("Step 11: Sequential filter for breaking big elements of size >", 300, "with Opening se =", 7), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=3)

        img_big_elements_beaked = break_big_elements(img_big_elements_beaked, size_th=400, se=10)
        # notebook_show(img_big_elements_beaked, ("Step 12: Sequential filter for breaking big elements of size >", 400, "with Opening se =", 10), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=3)

        # connecting small elements
        SE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        img_filtered_close = cv.morphologyEx(img_big_elements_beaked, cv.MORPH_CLOSE, SE)
        # notebook_show(img_big_elements_beaked, ("Step 13: Connecting small elements with Closing se =", 3), show=self.notebook, explanation_depth=self.explanation_depth, step_depth=3)

        return img_filtered_close
    
    

    # #-------------- added only to show the steps
    # def __sub_preprocessing_notebook(self, img, sub_preprocessing_step=0):
    #     gaussian_w = self.GAUSSIAN_W
    #     if self.GAUSSIAN_W%2==0:
    #         gaussian_w+=1
    #     img_res = cv.GaussianBlur(img, (gaussian_w, gaussian_w), 0)

    #     if sub_preprocessing_step > 0:
    #         img_res = gamma_correction(img_res, self.SUB_P_GAMMA_C)

    #     if sub_preprocessing_step > 1:
    #         _, img_res = cv.threshold(img_res, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    #     return img_res
    # #-------------- end added only to show the steps
