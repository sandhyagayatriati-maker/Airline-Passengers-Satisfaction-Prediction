from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load your trained model
with open("newmodel.pkl", "rb") as f:
    model = pickle.load(f)

# Columns used during training (after one-hot encoding)
model_columns = [
    'ID','Age', 'Flight Distance', 'Departure Delay', 'Arrival Delay',
    'Departure and Arrival Time Convenience', 'Ease of Online Booking',
    'Check-in Service', 'Online Boarding', 'Gate Location', 'On-board Service',
    'Seat Comfort', 'Leg Room Service', 'Cleanliness', 'Food and Drink',
    'In-flight Service', 'In-flight Wifi Service', 'In-flight Entertainment',
    'Baggage Handling',
    'Gender_Female', 'Gender_Male',
    'Customer Type_First-time', 'Customer Type_Returning',
    'Type of Travel_Business', 'Type of Travel_Personal',
    'Class_Business', 'Class_Economy', 'Class_Economy Plus'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input data from form
        input_data = {
            "Gender": request.form["Gender"],
            "Age": float(request.form["Age"]),
            "Customer Type": request.form["Customer Type"],
            "Type of Travel": request.form["Type of Travel"],
            "Class": request.form["Class"],
            "Flight Distance": float(request.form["Flight Distance"]),
            "Departure Delay": float(request.form["Departure Delay"]),
            "Arrival Delay": float(request.form["Arrival Delay"]),
            "Seat Comfort": float(request.form["Seat Comfort"]),
            "Cleanliness": float(request.form["Cleanliness"]),
            "Baggage Handling": float(request.form["Baggage Handling"]),
            "In-flight Service": float(request.form["In-flight Service"]),
            "Food and Drink": float(request.form["Food and Drink"]),
            "In-flight Wifi Service": float(request.form["In-flight Wifi Service"]),
            "In-flight Entertainment": float(request.form["In-flight Entertainment"]),
            "Gate Location": float(request.form["Gate Location"]),
            "Online Boarding": float(request.form["Online Boarding"]),
            "Ease of Online Booking": float(request.form["Ease of Online Booking"]),
            "Check-in Service": float(request.form["Check-in Service"]),
            "Leg Room Service": float(request.form["Leg Room Service"]),
            "Departure and Arrival Time Convenience": float(request.form["Departure and Arrival Time Convenience"])
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'])

        # Reindex to match training columns, missing columns filled with 0
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

        # Make prediction
        pred = model.predict(df_encoded)[0]
        pred_label = "Satisfied" if pred == 1 else "Neutral/Dissatisfied"
        probability = model.predict_proba(df_encoded)[0].max()

        return jsonify({
            "prediction": pred_label,
            "probability": round(float(probability), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)



