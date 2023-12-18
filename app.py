from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Mapping for encoding categorical variables
work_type_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
smoking_status_mapping = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data_to_predict = {
            'gender': request.form['gender'],
            'age': int(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'ever_married': request.form['ever_married'],
            'work_type': request.form['work_type'],
            'Residence_type': request.form['Residence_type'],
            'avg_glucose_level': float(request.form['avg_glucose_level']),
            'bmi': float(request.form['bmi']),
            'smoking_status': request.form['smoking_status']
        }

        # Prepare the input data for prediction
        data_to_predict_df = pd.DataFrame([data_to_predict])
        data_to_predict_df['gender'] = (data_to_predict_df['gender'] == 'Male').astype(int)
        data_to_predict_df['ever_married'] = (data_to_predict_df['ever_married'] == 'Yes').astype(int)
        data_to_predict_df['work_type'] = data_to_predict_df['work_type'].map(work_type_mapping)
        data_to_predict_df['Residence_type'] = (data_to_predict_df['Residence_type'] == 'Urban').astype(int)
        data_to_predict_df['smoking_status'] = data_to_predict_df['smoking_status'].map(smoking_status_mapping)

        # Make predictions using the loaded model
        prediction = loaded_model.predict(data_to_predict_df)
        
        # Return the prediction result
        return render_template('index.html', prediction_text=f'Prediction: {prediction[0]}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
