

import pathlib
import os

from .src.classifier import Classifier
from .src.functions import *
from .src.advanced_od_removal import AdvancedODRemoval


def get_base_folder():
    return str(pathlib.Path(__file__).parent.resolve()).replace(os.sep, "/") + "/"

MODEL_PATH_NAME = get_base_folder() + 'models/model_v0_train_val_feats.pkl'

class SexDetector:

    def test(self, img, img_gt_se=None, img_gt_od=None):
        # perfom detection
        classifier = Classifier()
        print(MODEL_PATH_NAME)
        classifier.load_model(MODEL_PATH_NAME)
        pred_y, prob_y, candidate_contours, rois = classifier.test(img)

        # obtain number of candidates and number of suspicious candidates
        filtered_contours = [contour for i, contour in enumerate(candidate_contours) if pred_y[i] == 1]
        nb_candidates = len(candidate_contours)
        nb_suspicious_candidates = len(filtered_contours)

        # draw the results on the image
        mask_od_advanced = AdvancedODRemoval().get_od_mask(img)
        img_show =  draw_contours(img, candidate_contours, thickness=5, color=(255,255,255))
        img_show = draw_contours(img_show, filtered_contours, thickness=5, color=(255,0,0))
        # also show the OD mask
        img_show = draw_mask(img_show, mask_od_advanced, mask_color=(0,0,0), thickness=5)

        # if available, draw the ground truth
        if img_gt_se is not None:
            img_show = draw_mask(img_show, img_gt_se, mask_color=(0,255,0), thickness=5)
        if img_gt_od is not None:
            img_show = draw_mask(img_show, img_gt_od, mask_color=(255,255,0), thickness=5)

        return {'image':img_show, 
                'nb_candidates': nb_candidates, 
                'nb_suspicious_candidates': nb_suspicious_candidates}


