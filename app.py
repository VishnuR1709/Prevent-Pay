import pickle
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

CATEGORIES = [
    'entertainment', 'food_dining', 'gas_transport', 'grocery_net',
    'grocery_pos', 'health_fitness', 'home', 'kids_pets',
    'misc_net', 'misc_pos', 'personal_care', 'shopping_net',
    'shopping_pos', 'travel'
]

STATES = [
    'AK','AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI',
    'IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN',
    'MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH',
    'OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA',
    'WI','WV','WY'
]

MODEL_READY = False
model = scaler = feature_columns = None

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    MODEL_READY = True
    print("Model loaded successfully.")
except FileNotFoundError as e:
    print(f"[WARNING] {e}")
    print("Run 'python save_model.py' to generate model artifacts, then restart.")


@app.route('/')
def index():
    return render_template(
        'index.html',
        model_ready=MODEL_READY,
        categories=CATEGORIES,
        states=STATES
    )


@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_READY:
        return jsonify({'error': 'Model not loaded. Run save_model.py first, then restart the server.'}), 503

    try:
        data = request.get_json(force=True)

        required = ['cc_num','amt','trans_datetime','dob','gender',
                    'lat','long','city_pop','merch_lat','merch_long',
                    'category','state']
        missing = [f for f in required if f not in data or str(data[f]).strip() == '']
        if missing:
            return jsonify({'error': f"Missing required fields: {', '.join(missing)}"}), 400

        amt = float(data['amt'])
        if amt <= 0:
            return jsonify({'error': 'Transaction amount must be greater than 0.'}), 400

        lat  = float(data['lat'])
        long = float(data['long'])
        if not (-90 <= lat <= 90):
            return jsonify({'error': 'Cardholder latitude must be between -90 and 90.'}), 400
        if not (-180 <= long <= 180):
            return jsonify({'error': 'Cardholder longitude must be between -180 and 180.'}), 400

        category = data['category']
        state    = data['state']
        if category not in CATEGORIES:
            return jsonify({'error': f"Invalid category: {category}"}), 400
        if state not in STATES:
            return jsonify({'error': f"Invalid state: {state}"}), 400

        dt = datetime.fromisoformat(data['trans_datetime'])
        trans_hour  = dt.hour
        trans_day   = dt.day
        trans_month = dt.month
        trans_year  = dt.year
        unix_time   = int(dt.timestamp())

        dob        = datetime.fromisoformat(data['dob'])
        birth_year = dob.year
        age        = trans_year - birth_year

        gender  = 1 if data['gender'] == 'M' else 0
        cc_num  = int(data['cc_num'])
        city_pop   = int(data['city_pop'])
        merch_lat  = float(data['merch_lat'])
        merch_long = float(data['merch_long'])

        row = {col: 0 for col in feature_columns}
        row['cc_num']     = cc_num
        row['amt']        = amt
        row['gender']     = gender
        row['lat']        = lat
        row['long']       = long
        row['city_pop']   = city_pop
        row['unix_time']  = unix_time
        row['merch_lat']  = merch_lat
        row['merch_long'] = merch_long
        row['trans_hour']  = trans_hour
        row['trans_day']   = trans_day
        row['trans_month'] = trans_month
        row['trans_year']  = trans_year
        row['birth_year']  = birth_year
        row['age']         = age

        cat_col   = f'category_{category}'
        state_col = f'state_{state}'
        if cat_col in row:
            row[cat_col] = 1
        if state_col in row:
            row[state_col] = 1

        X = np.array([[row[col] for col in feature_columns]])
        X_scaled = scaler.transform(X)

        prediction = int(model.predict(X_scaled)[0])
        proba      = model.predict_proba(X_scaled)[0]
        confidence = round(float(max(proba)) * 100, 2)

        return jsonify({
            'prediction': prediction,
            'label':      'Fraudulent' if prediction == 1 else 'Legitimate',
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
