
import pandas as pd
from sklearn.svm import SVC
import joblib

from .functions import *
from .advanced_sex_detection import AdvancedSExDetection
from .features_engeneering import SExFeaturesEngeneering
from .features_extractor import SExFeaturesExtractor


class Classifier:

    def __init__(self):
        self.feat_engeneering = SExFeaturesEngeneering()
        self.svm = SVC(kernel="poly", gamma='auto', class_weight='balanced', probability=True)
        self.trained = False

    def train_from_features(self, x, y):
        x = self.feat_engeneering.fit_transform(x, y)

        # train model
        self.svm.fit(x, y)
        pred_y = self.svm.predict(x)
        prob_y = self.svm.predict_proba(x)

        self.trained = True
        return pred_y, prob_y


    def test(self, img):
        if not self.trained:
            raise Exception('Model not trained')
        
        print('-------- Test mode --------')
        print('step 1. Obtain candidate contours usin AdvancedSExDetection')
        candidate_contours, rois = AdvancedSExDetection().exudates_detection(img)
        print('step 2. Extract features from candidate contours')
        x =  SExFeaturesExtractor().extact_features(img, candidate_contours, rois)
        x = self.feat_engeneering.transform(x)
        print('step 3. Predict using SVM')
        pred_y = self.svm.predict(x)
        prob_y = self.svm.predict_proba(x)

        return pred_y, prob_y, candidate_contours, rois
        

    def save_model(self, filepath):
        model_data = {
            'svm': self.svm,
            'feat_engeneering': self.feat_engeneering,  # Incluye el feature engineering
        }
        joblib.dump(model_data, filepath)
        print(f'Model saved to {filepath}')


    # Función para cargar el modelo y el feature engineering
    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.svm = model_data['svm']
        self.feat_engeneering = model_data['feat_engeneering']  # Recarga el feature engineering
        self.trained = True
        print(f'Model loaded from {filepath}')