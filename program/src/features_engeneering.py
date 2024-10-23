
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

class SExFeaturesEngeneering:

    def __init__(self):
        self.scaler = None
        self.selector = None
        self.zero_columns = None

    def fit_transform(self, features, labels, nb_features=-1):

        # step 1: satndarize features
        self.scaler = StandardScaler()
        feat_standarized = self.scaler.fit_transform(features)

        # step 2: remove null colums
        self.zero_columns = np.all(feat_standarized == 0, axis=0)
        feat_standarized = feat_standarized[:, ~self.zero_columns]

        # step 3: feature selection
        if nb_features > 0:
            self.selector = SelectKBest(f_classif, k=nb_features)
            feat_standarized = self.selector.fit_transform(feat_standarized, labels)

        return feat_standarized


    def transform(self, features):
        feat_standarized = self.scaler.transform(features)
        feat_standarized = feat_standarized[:, ~self.zero_columns]
        if self.selector is not None:
            feat_standarized = self.selector.transform(feat_standarized)

        return feat_standarized







# --------- same as above but with PCA ------------

# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.decomposition import PCA

# class SExFeaturesEngeneering:

#     def __init__(self):
#         self.scaler = None
#         self.selector = None
#         self.zero_columns = None
#         self.pca = None

#     def fit_transform(self, features, labels, nb_features, nb_pca_components=-1):

#         # Step 1: Estándarizar las características
#         self.scaler = StandardScaler()
#         feat_standarized = self.scaler.fit_transform(features)

#         # Step 2: Eliminar columnas nulas
#         self.zero_columns = np.all(feat_standarized == 0, axis=0)
#         feat_standarized = feat_standarized[:, ~self.zero_columns]

#         # Step 3: Selección de características
#         if nb_features > 0:
#             self.selector = SelectKBest(f_classif, k=nb_features)
#             feat_standarized = self.selector.fit_transform(feat_standarized, labels)

#         # Step 4: Aplicar PCA si se requiere
#         if nb_pca_components != 0:
#             if nb_pca_components > 0:
#                 self.pca = PCA(n_components=nb_pca_components)
#             else:
#                 self.pca = PCA()
#             feat_standarized = self.pca.fit_transform(feat_standarized)

#         return feat_standarized

#     def transform(self, features):
#         # Step 1: Estándarizar las características
#         feat_standarized = self.scaler.transform(features)

#         # Step 2: Eliminar columnas nulas
#         feat_standarized = feat_standarized[:, ~self.zero_columns]

#         # Step 3: Aplicar la selección de características si es necesario
#         if self.selector is not None:
#             feat_standarized = self.selector.transform(feat_standarized)

#         # Step 4: Aplicar PCA si se requiere
#         if self.pca is not None:
#             feat_standarized = self.pca.transform(feat_standarized)

#         return feat_standarized
