from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd  # Import Pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

app = Flask(__name__)

# Load and prepare data
customers = pd.read_csv('Ecommerce Customers')
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Train Linear Regression model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Train KNN model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Route for the home page (form)
@app.route('/')
def home():
    return render_template('index.html')  # This renders the home page (form)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract JSON data from the request
            data = request.get_json()

            # Get values from the request
            avg_session = float(data['Avg'])
            time_on_app = float(data['Toapp'])
            time_on_website = float(data['Toweb'])
            length_of_membership = float(data['Lomember'])

            # Prepare the input features for prediction
            features = np.array([[avg_session, time_on_app, time_on_website, length_of_membership]])

            # Convert the NumPy array to a DataFrame with feature names
            feature_names = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
            features_df = pd.DataFrame(features, columns=feature_names)

            # Get predictions from both models
            prediction_lm = lm.predict(features_df)[0]
            prediction_knn = knn.predict(features_df)[0]

            # Format predictions to 2 decimal places
            formatted_lm = round(prediction_lm, 2)
            formatted_knn = round(prediction_knn, 2)

            # Return predictions as a JSON response
            return jsonify({
                'Linear Regression Prediction': formatted_lm,
                'KNN Prediction': formatted_knn
            })
        except ValueError:
            return jsonify({"error": "Invalid input data. Please enter valid numerical values."}), 400

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
