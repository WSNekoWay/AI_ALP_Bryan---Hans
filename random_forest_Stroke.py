import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def main():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')

    f = df[df['gender'] != 'Other']

    median_bmi = df['bmi'].median()
    df['bmi'].fillna(median_bmi, inplace=True)
    df.dropna(inplace=True)

    label_encoder = LabelEncoder()

    df['gender'] = label_encoder.fit_transform(df['gender'])
    df['ever_married'] = label_encoder.fit_transform(df['ever_married'])
    df['work_type'] = label_encoder.fit_transform(df['work_type'])
    df['Residence_type'] = label_encoder.fit_transform(df['Residence_type'])
    df['smoking_status'] = label_encoder.fit_transform(df['smoking_status'])

    X = df.drop(['id', 'stroke'], axis=1)
    y = df['stroke']

    numerical_cols = ['age', 'avg_glucose_level', 'bmi']

    smote = SMOTE(random_state=42, sampling_strategy = 'minority')
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols]) 
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    best_accuracy = 0
    best_params = None

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    }

    for criterion in param_grid['criterion']:
    
        model = RandomForestClassifier(random_state=42, criterion=criterion, n_jobs=-1)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_weighted', cv=5)

        grid_search.fit(X_train, y_train)
        
        if grid_search.best_score_ > best_accuracy:
            best_accuracy = grid_search.best_score_
            best_params = grid_search.best_params_

    print("Best Hyperparameters:", best_params)

    final_model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'], criterion=best_params['criterion'], class_weight=['class_weight'], random_state=42)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision:.4f}")

    recall = recall_score(y_test, y_pred)
    print(f"Recall: {recall:.4f}")

    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")

    class_report = classification_report(y_test, y_pred)
    print("Classification Report:\n", class_report)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    print(f"AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()