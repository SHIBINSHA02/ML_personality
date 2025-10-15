# model.py
# model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

class PersonalityModel:
    def __init__(self, data_path='data.csv'):
        self.data_path = data_path
        self.model = None
        self.le_stage = None
        self.le_drained = None
        self.le_personality = None

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)


        numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                        'Friends_circle_size', 'Post_frequency']
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())


        categorical_cols = ['Stage_fear', 'Drained_after_socializing']
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        self.le_stage = LabelEncoder()
        df['Stage_fear'] = self.le_stage.fit_transform(df['Stage_fear'])

        self.le_drained = LabelEncoder()
        df['Drained_after_socializing'] = self.le_drained.fit_transform(df['Drained_after_socializing'])

        self.le_personality = LabelEncoder()
        df['Personality'] = self.le_personality.fit_transform(df['Personality'])

        return df

    def train_model(self):
        df = self.load_and_preprocess_data()
        X = df[['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 
                'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']]
        y = df['Personality']

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

    def predict(self, input_data):
        """
        input_data: dict with keys matching the features
        """

        stage_fear_enc = self.le_stage.transform([input_data['Stage_fear']])[0]
        drained_enc = self.le_drained.transform([input_data['Drained_after_socializing']])[0]

        data_array = np.array([[input_data['Time_spent_Alone'], stage_fear_enc,
                                input_data['Social_event_attendance'], input_data['Going_outside'],
                                drained_enc, input_data['Friends_circle_size'], input_data['Post_frequency']]])

        pred = self.model.predict(data_array)[0]
        pred_label = self.le_personality.inverse_transform([pred])[0]
        return pred_label

    def get_coefficients(self):
        return dict(zip(self.model.coef_[0], ['Time_spent_Alone','Stage_fear','Social_event_attendance',
                                             'Going_outside','Drained_after_socializing','Friends_circle_size',
                                             'Post_frequency']))
