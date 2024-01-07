import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
import warnings
import pickle

warnings.filterwarnings("ignore")

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

df = df[df['gender'] != 'Other']

imputer = KNNImputer(n_neighbors=5)
df['bmi'] = imputer.fit_transform(df[['bmi']])
mode_smoking_status = df['smoking_status'].mode()[0]
df['smoking_status'].replace('Unknown', mode_smoking_status, inplace=True)

df.dropna(inplace=True)

label_encoder = LabelEncoder()

df['gender'] = label_encoder.fit_transform(df['gender'])
df['ever_married'] = label_encoder.fit_transform(df['ever_married'])
df['work_type'] = label_encoder.fit_transform(df['work_type'])
df['Residence_type'] = label_encoder.fit_transform(df['Residence_type'])
df['smoking_status'] = label_encoder.fit_transform(df['smoking_status'])

numerical_cols = ['age', 'avg_glucose_level', 'bmi']

Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

smote  = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

best_params = None

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced']
}
 
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_weighted', cv=10, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

final_model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'], criterion=best_params['criterion'], class_weight=best_params['class_weight'], random_state=42)
final_model.fit(X_train, y_train)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(final_model, model_file)

loaded_model = pickle.load(open('model.pkl', 'rb'))

data_to_predict = {
    'gender': 'Male',
    'age': 55,
    'hypertension': 1,
    'heart_disease': 0,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 85.5,
    'bmi': 28.0,
    'smoking_status': 'formerly smoked'
}


data_to_predict_df = pd.DataFrame([data_to_predict])
gender_mapping = {'Male': 1, 'Female': 0}
ever_married_mapping = {'Yes': 1, 'No':0}
work_type_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
residence_type_mapping = {'Urban': 1, 'Rural': 0}
smoking_status_mapping = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}

data_to_predict_df['gender'] = data_to_predict_df['gender'].map(gender_mapping)
data_to_predict_df['ever_married'] = data_to_predict_df['ever_married'].map(ever_married_mapping)
data_to_predict_df['work_type'] = data_to_predict_df['work_type'].map(work_type_mapping)
data_to_predict_df['Residence_type'] = data_to_predict_df['Residence_type'].map(residence_type_mapping)
data_to_predict_df['smoking_status'] = data_to_predict_df['smoking_status'].map(smoking_status_mapping)

prediction = loaded_model.predict(data_to_predict_df)
print("Prediction:", prediction[0])
