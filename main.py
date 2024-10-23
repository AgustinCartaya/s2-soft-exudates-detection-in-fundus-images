
import cv2 as cv

from program.sex_detector import SexDetector
from program import model_creator_from_features
from program.src import functions as fc
import security_check


if __name__ == '__main__':
    # ---- train the model from the features
    # model_creator_from_features.save_model(use_val_fatures=True)

    # ---- test the model
    img_path_name = "C:/Users/Agustin/portfolio/medical_imaging_projects/s2_soft_exudates_detection_in_fundus_images/code/program/data/images/training/IDRiD_13.jpg"
    img_gt_od_path_name = "C:/Users/Agustin/portfolio/medical_imaging_projects/s2_soft_exudates_detection_in_fundus_images/code/program/data/groundtruths/groundtruths_OD/training/IDRiD_13_OD.tif"
    img_gt_se_path_name = "C:/Users/Agustin/portfolio/medical_imaging_projects/s2_soft_exudates_detection_in_fundus_images/code/program/data/groundtruths/groundtruths_SE/training/IDRiD_13_SE.tif"

    img = cv.imread(img_path_name, cv.IMREAD_COLOR)
    img_gt_se = cv.imread(img_gt_se_path_name, cv.IMREAD_GRAYSCALE)
    img_gt_od = cv.imread(img_gt_od_path_name, cv.IMREAD_GRAYSCALE)

    security_message, is_secure = security_check.check_image(img)
    if not is_secure:
        raise Exception('Security check failed: {}'.format(security_message))
    
    result_dict = SexDetector().test(img, img_gt_se, img_gt_od)
    print("Number of SE detected before SVM:", result_dict["nb_candidates"])
    print("Number of SE detected after SVM:", result_dict["nb_suspicious_candidates"])

    fc.notebook_show(result_dict["image"], "Original image", show=True)
