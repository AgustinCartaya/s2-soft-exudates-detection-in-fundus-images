
import skimage
import scipy
import glrlm
from tqdm import tqdm

from .functions import *


class SExFeaturesExtractor:

    def extact_features(self, img, contours, rois, roi_enhancement='none'):
        features_list = []

        for i in tqdm(range(len(rois))):
            mask = create_contours_mask(contours, img.shape, i)
            mask_roi = get_roi(mask, rois[i])

            img_roi = get_roi(img, rois[i])
            img_roi_masked = apply_mask(img_roi, mask_roi, -1)

            features = self.extract_features_from_roi(img_roi_masked, roi_enhancement)
            features_list.append(features)
        return features_list


    def extract_features_from_roi(self, img_roi_masked, roi_enhancement):
        img_enhanced = self.enhance_image(img_roi_masked, roi_enhancement) 

        roi_fourier = self.calculate_fourier_descriptors(img_enhanced) # 7 0-6
        roi_shape = self.calculate_shape_descriptors(img_enhanced) # 6 7-12
        roi_hu = self.calculate_hu_moments(img_enhanced) # 7 13-19
        roi2glcm = self.extract_glcm_features(img_enhanced) #37 20-56
        roi2lbp = self.extract_lbp_features(img_enhanced) #10 57-66 (255 57-311)
        roi_color = self.calculate_color_descriptors(img_roi_masked) # 21 67-87 (255 312-332)
        roi_glrlm = self.extract_features_glrlm(img_enhanced) # 5 88-92 (255 333-333)

        all_features = np.concatenate((roi_fourier, roi_shape, roi_hu, roi2glcm, roi2lbp, roi_color, roi_glrlm))

        return all_features


    def extract_features_glrlm(self, roi):
        app = glrlm.GLRLM()
        glrlm_feat = app.get_features(roi, 8)

        return glrlm_feat.Features

    def calculate_haralick_features(self, glcm):
        # Calculate Haralick texture features from GLCM
        haralick_features = skimage.feature.graycoprops(glcm)

        # Compute the mean of each Haralick feature
        haralick_mean = np.mean(haralick_features, axis=0)

        return haralick_mean

    def calculate_cluster_prominence(self, glcm):
        p = np.indices(glcm.shape)[0]
        q = np.indices(glcm.shape)[1]

        mean_row = np.sum(p * glcm) / np.sum(glcm)
        mean_col = np.sum(q * glcm) / np.sum(glcm)

        cluster_prominence = np.sum(((p + q - mean_row - mean_col) ** 4) * glcm) / np.sum(glcm) ** 2
        return cluster_prominence

    def calculate_cluster_shade(self, glcm):
        # Calculate cluster shade
        p = np.indices(glcm.shape)[0]
        q = np.indices(glcm.shape)[1]
        mean_row = np.sum(p * glcm) / np.sum(glcm)
        mean_col = np.sum(q * glcm) / np.sum(glcm)
        cluster_shade = np.sum(((p + q - mean_row - mean_col) ** 3) * glcm) / np.sum(glcm) ** 2
        return cluster_shade

    def calculate_max_probability(self, glcm):
        # Calculate max probability
        max_probability = np.max(glcm) / np.sum(glcm)
        return max_probability

    def calculate_sum_average(self, glcm):
        # Calculate sum average
        p = np.indices(glcm.shape)[0]
        q = np.indices(glcm.shape)[1]
        sum_average = np.sum((p + q) * glcm) / np.sum(glcm)
        return sum_average

    def calculate_sum_variance(self, glcm):
        # Calculate sum variance
        p = np.indices(glcm.shape)[0]
        q = np.indices(glcm.shape)[1]
        sum_average = self.calculate_sum_average(glcm)
        sum_variance = np.sum(((p + q) - sum_average) ** 2 * glcm) / np.sum(glcm)
        return sum_variance

    def calculate_sum_entropy(self, glcm):
        # Calculate sum entropy
        glcm_normalized = glcm / np.sum(glcm)
        sum_entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + 1e-10))
        return sum_entropy

    def calculate_difference_variance(self, glcm):
        # Calculate difference variance
        p = np.indices(glcm.shape)[0]
        q = np.indices(glcm.shape)[1]
        difference_variance = np.sum(((p - q) ** 2) * glcm) / np.sum(glcm)
        return difference_variance

    def calculate_difference_entropy(self, glcm):
        # Calculate difference entropy
        glcm_normalized = glcm / np.sum(glcm)
        difference_entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + 1e-10))
        return difference_entropy

    def extract_glcm_features(self, roi):
        graycom = skimage.feature.graycomatrix(roi, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)

        contrast = skimage.feature.graycoprops(graycom, 'contrast')
        dissimilarity = skimage.feature.graycoprops(graycom, 'dissimilarity')
        homogeneity = skimage.feature.graycoprops(graycom, 'homogeneity')
        energy = skimage.feature.graycoprops(graycom, 'energy')
        correlation = skimage.feature.graycoprops(graycom, 'correlation')
        asm = skimage.feature.graycoprops(graycom, 'ASM')

        cluster_prominence = self.calculate_cluster_prominence(graycom)
        cluster_shade = self.calculate_cluster_shade(graycom)
        max_probability = self.calculate_max_probability(graycom)
        sum_average = self.calculate_sum_average(graycom)
        sum_variance = self.calculate_sum_variance(graycom)
        sum_entropy = self.calculate_sum_entropy(graycom)
        difference_variance = self.calculate_difference_variance(graycom)
        difference_entropy = self.calculate_difference_entropy(graycom)
        haralick = self.calculate_haralick_features(graycom)

        lacunarity = 1 - (contrast.var() / contrast.mean() ** 2)
        # print(contrast.shape)

        features = np.concatenate([
            contrast.ravel(),
            dissimilarity.ravel(),
            homogeneity.ravel(),
            energy.ravel(),
            correlation.ravel(),
            asm.ravel(),
            lacunarity.ravel(),
            cluster_prominence.ravel(),
            cluster_shade.ravel(),
            max_probability.ravel(),
            sum_average.ravel(),
            sum_variance.ravel(),
            sum_entropy.ravel(),
            difference_variance.ravel(),
            difference_entropy.ravel(),
            haralick.ravel()
        ])

        return features


    def extract_lbp_features(self, roi):
        lbp = skimage.feature.local_binary_pattern(roi, 8, 1, method='uniform')
        num_patterns = int(lbp.max() + 1)  

        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_patterns + 1), range=(0, num_patterns), density=True)
        return hist

    def calculate_fourier_descriptors(self, img):
        f = np.fft.fft2(img)

        fshift = np.fft.fftshift(f)

        magnitude_spectrum = np.abs(fshift)
        magnitude_spectrum /= np.sum(magnitude_spectrum) 

        phase_spectrum = np.angle(fshift)
        phase_spectrum /= (2 * np.pi)  

        total_energy = np.sum(magnitude_spectrum**2)

        cumulative_energy = np.cumsum(magnitude_spectrum**2) / total_energy

        mean_amplitude = np.mean(magnitude_spectrum)
        std_amplitude = np.std(magnitude_spectrum)
        max_amplitude = np.max(magnitude_spectrum)
        mean_phase = np.mean(phase_spectrum)
        std_phase = np.std(phase_spectrum)
        max_phase = np.max(phase_spectrum)
        energy_ratio = cumulative_energy[10] 

        return np.array([mean_amplitude, std_amplitude, max_amplitude, mean_phase, std_phase, max_phase, energy_ratio])

    def calculate_shape_descriptors(self, img):
        _, th = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, _ = cv.findContours(th, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        num_regions, _, _, _ = cv.connectedComponentsWithStats(th)
        sodality = num_regions - 1

        for contour in contours:
            area = cv.contourArea(contour)
            perimeter = cv.arcLength(contour, True)

            if perimeter == 0:
                circularity = 0
            else:
                circularity = (4 * np.pi * area) / (perimeter**2)

            _, _, w, h = cv.boundingRect(contour)
            aspect_ratio = float(w) / h

            rect_area = w * h    
            extent = area / rect_area


        return np.array([area, perimeter, circularity, aspect_ratio, sodality, extent])


    def calculate_hu_moments(self, img):
        image_array = np.array(img, dtype=np.float32)

        covariance_matrix = np.cov(image_array, rowvar=False)

        # Calcular los momentos centrales
        moments = cv.moments(covariance_matrix)

        # Calcular los momentos de Hu
        hu_moments = cv.HuMoments(moments)

        return np.array([hu_moments[0][0], hu_moments[1][0], hu_moments[2][0], hu_moments[3][0], hu_moments[4][0], hu_moments[5][0], hu_moments[6][0]])


    def calculate_color_descriptors(self, img):
        img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        img_lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        intensity_values = img_gray.flatten()


        red_channel = cv.extractChannel(img, 0)
        green_channel = cv.extractChannel(img, 1)
        blue_channel = cv.extractChannel(img, 2)

        std_red = np.std(red_channel)
        std_green = np.std(green_channel)
        std_blue = np.std(blue_channel)
        mean_red = np.mean(red_channel)
        mean_green = np.mean(green_channel)
        mean_blue = np.mean(blue_channel)

        h_channel = cv.extractChannel(img_hsv, 0)
        s_channel = cv.extractChannel(img_hsv, 1)
        v_channel = cv.extractChannel(img_hsv, 2)

        std_h = np.std(h_channel)
        std_s = np.std(s_channel)
        std_v = np.std(v_channel)
        mean_h = np.mean(h_channel)
        mean_s = np.mean(s_channel)
        mean_v = np.mean(v_channel)

        l_channel = cv.extractChannel(img_lab, 0)
        a_channel = cv.extractChannel(img_lab, 1)
        b_channel = cv.extractChannel(img_lab, 2)

        std_l = np.std(l_channel)
        std_a = np.std(a_channel)
        std_b = np.std(b_channel)
        mean_l = np.mean(l_channel)
        mean_a = np.mean(a_channel)
        mean_b = np.mean(b_channel)

        variance = np.var(img_gray)

        skewness = scipy.stats.skew(intensity_values)
        kurt = scipy.stats.kurtosis(intensity_values)


        return np.array([std_red, std_green, std_blue, std_h, std_s, std_v, std_l, std_a, std_b,
                        mean_red, mean_green, mean_blue, mean_h, mean_s, mean_v, mean_l, mean_a, mean_b, variance, skewness, kurt])


    def enhance_image(self, img, roi_enhancement):
        if roi_enhancement == 'none':
            img_enhanced = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        else:
            green = cv.extractChannel(img, 1) 

            if roi_enhancement == 'green':
                img_enhanced = green
                
            elif roi_enhancement == 'median_blur':
                img_enhanced = cv.medianBlur(green, 3)  

            elif roi_enhancement == 'clahe':
                clahe = cv.createCLAHE(clipLimit=14)
                img_enhanced = clahe.apply(green)

            elif roi_enhancement == 'gamma':
                img_enhanced = gamma_correction(green, 0.8)

            elif roi_enhancement == 'equalize':
                img_enhanced = cv.equalizeHist(green)

            elif roi_enhancement == 'gaussian_blur':
                img_enhanced = cv.GaussianBlur(green, (3, 3), 0)

            elif roi_enhancement == 'normalize':
                img_enhanced = cv.normalize(green, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

            elif roi_enhancement == 'denoising':
                img_enhanced = cv.fastNlMeansDenoising(green, h=10, searchWindowSize=21)

        return img_enhanced


