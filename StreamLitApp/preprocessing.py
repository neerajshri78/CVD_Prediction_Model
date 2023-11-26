import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from kmodes.kmodes import KModes
import pickle

class DataPreprocessor:
    def __init__(self):
        self.le = preprocessing.LabelEncoder()
        with open('../CVD_Modelling/Model_Analysis/Model_Pickle/female_clusters.pkl', 'rb') as file:
            self.kmodes_female = pickle.load(file)

        with open('../CVD_Modelling/Model_Analysis/Model_Pickle/male_clusters.pkl', 'rb') as file:
            self.kmodes_male = pickle.load(file)

    def preprocess_data(self, data):
        # Calculate 'age_bin' based on provided bin_range and bin_labels
        bin_range = [0, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        bin_labels = [f'{bin_start}-{bin_end}' for bin_start, bin_end in zip(bin_range[:-1], bin_range[1:])]
        data['age_bin'] = pd.cut(data['age'], bins=bin_range, labels=bin_labels)
        data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)

        # Calculate 'BMI_Class' based on BMI calculations
        bmi_ratings = []

        for row in data['bmi']:
            if row < 18.5:
                bmi_ratings.append(1)  # Underweight
            elif 18.5 <= row < 24.9:
                bmi_ratings.append(2)  # Normal Weight
            elif 24.9 <= row < 29.9:
                bmi_ratings.append(3)  # Overweight
            elif 29.9 <= row < 34.9:
                bmi_ratings.append(4)  # Class Obesity 1
            elif 34.9 <= row < 39.9:
                bmi_ratings.append(5)  # Class Obesity 2
            elif 39.9 <= row < 49.9:
                bmi_ratings.append(6)  # Class Obesity 3
            else:
                bmi_ratings.append('Error')
        data['BMI_Class'] = bmi_ratings

        # Calculate 'MAP' and 'MAP_Class' based on provided logic
        data['MAP'] = ((2 * data['ap_lo']) + data['ap_hi']) / 3
        map_values = []
        for row in data['MAP']:
            if row < 69.9:
                map_values.append(1)  # Low
            elif 70 <= row < 79.9:
                map_values.append(2)  # Normal
            elif 79.9 <= row < 89.9:
                map_values.append(3)  # Normal
            elif 89.9 <= row < 99.9:
                map_values.append(4)  # Normal
            elif 99.9 <= row < 109.9:
                map_values.append(5)  # High
            elif 109.9 <= row < 119.9:
                map_values.append(6)  # Normal
            else:
                map_values.append(7)
        data['MAP_Class'] = map_values

        # Drop the original columns that were used to create the features
        data = data.drop(['age', 'weight', 'height', 'ap_hi', 'ap_lo', 'bmi'], axis=1)

        return data

    def preprocess_and_cluster(self, data):
        df_cat = self.preprocess_data(data)

        # Encode categorical features using LabelEncoder
        df_cat_encoded = df_cat.apply(self.le.fit_transform)

        # Separate data by gender
        # df_male = df_cat_encoded[df_cat_encoded['gender'] == 0]
        # df_female = df_cat_encoded[df_cat_encoded['gender'] == 1]

        gender = df_cat_encoded['gender'].iloc[0]

        if gender == 0:
            clusters = self.kmodes_male.predict(df_cat_encoded)
            df_cat_encoded.insert(0, 'Cluster', clusters, True)
        elif gender == 1:
            clusters = self.kmodes_female.predict(df_cat_encoded)
            df_cat_encoded.insert(0, 'Cluster', clusters, True)
        else:
            raise ValueError("Invalid gender value. Expected 0 for male or 1 for female.")

        #
        #
        # # Apply K-Modes clustering to each gender-specific dataset
        # clusters_huang_1 = self.kmodes.fit_predict(df_female)
        # df_female.insert(0, 'Cluster', clusters_huang_1, True)
        #
        # clusters_huang_2 = self.kmodes.fit_predict(df_male)
        # df_male.insert(0, 'Cluster', clusters_huang_2, True)

        # Concatenate the results
        df_clusters = df_cat_encoded

        return df_clusters[
            ['Cluster', 'gender', 'age_bin', 'BMI_Class', 'MAP_Class', 'cholesterol', 'gluc', 'smoke', 'active']]

