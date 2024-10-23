
import pathlib
import os
import pandas as pd

from .src.classifier import Classifier


def get_base_folder():
    return str(pathlib.Path(__file__).parent.resolve()).replace(os.sep, "/") + "/"

FEATURES_TRAIN_PATH_NAME = get_base_folder() + "features/feat_training_complete.csv"
FEATURES_VAL_PATH_NAME = get_base_folder() + "features/feat_training_complete.csv"

MODEL_PATH_NAME = get_base_folder() + "models/model_v0.pkl"

def save_model(use_val_fatures=False):
    train_features = pd.read_csv(FEATURES_TRAIN_PATH_NAME)

    # load features
    if use_val_fatures:
        val_features = pd.read_csv(FEATURES_VAL_PATH_NAME)
        train_features = pd.concat([train_features, val_features])

    x = train_features.iloc[:, 2:].values
    y = train_features.iloc[:, 1].values

    sex_detector = Classifier()
    sex_detector.train_from_features(x, y)
    sex_detector.save_model(MODEL_PATH_NAME)


# you can't run this code here because the model wil be save with the actual path modules and the path will be different in the main
# run this code in the main