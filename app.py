from flask import Flask, request, jsonify

app = Flask(__name__)

# Simple function to predict loan approval based on some input features
def predict_loan_approval(age, income, credit_score):
    # This is a mock prediction logic
    # Normally, you would load a trained model and use it to predict, but here we use simple rules
    if credit_score > 700 and income > 3000 and age > 18:
        return "Approved"
    else:
        return "Denied"

@app.route('/')
def home():
    return "Welcome to the Loan Prediction App!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json()

    # Extract features from the input data
    age = data['age']
    income = data['income']
    credit_score = data['credit_score']

    # Call the prediction function
    prediction = predict_loan_approval(age, income, credit_score)

    # Return the prediction result as JSON
    return jsonify({'loan_status': prediction})

if __name__ == '__main__':
    app.run(debug=True)
