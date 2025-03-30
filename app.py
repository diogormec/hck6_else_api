import os
import json
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    CharField, Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = CharField(unique=True)
    observation = TextField()
    prediction = IntegerField()
    true_value = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model

with open('columns.json') as fh:
    columns = json.load(fh)

with open('pipeline.pickle', 'rb') as fh:
    pipeline = pickle.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


def clean_dollar_fields(data):
    """Clean dollar symbols and commas from monetary fields."""
    if 'Total Charges' in data and isinstance(data['Total Charges'], str):
        data['Total Charges'] = float(data['Total Charges'].replace('$', '').replace(',', ''))
    if 'Total Costs' in data and isinstance(data['Total Costs'], str):
        data['Total Costs'] = float(data['Total Costs'].replace('$', '').replace(',', ''))
    return data


def attempt_predict(request):
    """
    Produce prediction for hospital length of stay.
    
    Inputs:
        request: dictionary with patient data including observation_id
     
    Returns: A dictionary with the observation_id and prediction or an error
    """
    observation_id = request.get("observation_id", None)

    # Validate observation_id
    if not observation_id:
        return {
            "observation_id": None,
            "error": "Missing observation_id"
        }
    
    try:
        # Extract the data part (remove observation_id from input data)
        data = {k: v for k, v in request.items() if k != "observation_id"}
        
        # Clean monetary fields if needed
        data = clean_dollar_fields(data)
        
        # Filter to only include columns expected by the model
        model_data = {}
        for col in columns:
            if col in data:
                model_data[col] = data[col]
        
        # Check if we have all required columns
        if len(model_data) != len(columns):
            missing = set(columns) - set(model_data.keys())
            if missing:
                return {
                    "observation_id": observation_id,
                    "error": f"Missing required columns: {', '.join(missing)}"
                }
        
        # Convert to DataFrame
        obs = pd.DataFrame([model_data])
        
        # Convert data types according to the model expectations
        for col, dtype in dtypes.items():
            if col in obs.columns:
                try:
                    obs[col] = obs[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert {col} to {dtype}: {e}")
        
        # Generate the prediction
        prediction = pipeline.predict(obs)[0]
        
        # Format the response exactly as specified
        response = {
            "observation_id": observation_id,
            "prediction": str(int(prediction))  # Convert to string as per requirements
        }
        return response
    
    except Exception as e:
        # Handle errors by returning an error message
        print(f"Error during prediction: {str(e)}")
        return {
            "observation_id": observation_id,
            "error": f"Error during prediction: {str(e)}"
        }


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to handle prediction requests.
    Receives patient data, predicts length of stay, and returns observation_id and prediction.
    """
    # Deserialize the JSON payload from the request
    try:
        obs_dict = request.get_json()
    except Exception as e:
        return jsonify({"error": "Invalid JSON"}), 400

    result = attempt_predict(obs_dict)
    
    if "error" in result:
        return jsonify(result), 400
    
    # Store the prediction in the database
    try:
        p = Prediction.create(
            observation_id=result['observation_id'],
            prediction=int(result['prediction']),
            observation=json.dumps(obs_dict)
        )
        p.save()
    except IntegrityError:
        # Handle duplicate observation_id
        DB.rollback()
        try:
            # Update existing record
            p = Prediction.get(Prediction.observation_id == result['observation_id'])
            p.prediction = int(result['prediction'])
            p.observation = json.dumps(obs_dict)
            p.save()
        except Exception as e:
            print(f"Error updating existing record: {e}")
    except Exception as e:
        print(f"Error saving prediction: {e}")
    
    return jsonify(result), 200


@app.route('/update', methods=['POST'])
def update():
    """
    Endpoint to update with actual length of stay.
    Receives observation_id and true_value, updates database, and returns the same.
    """
    try:
        obs = request.get_json()
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    # Validate input format
    if not obs or 'observation_id' not in obs or 'true_value' not in obs:
        return jsonify({"error": "Invalid request: 'observation_id' and 'true_value' are required"}), 400

    try:
        # Get the observation_id and true_value
        observation_id = obs['observation_id']
        true_value = obs['true_value']
        
        # Try to convert true_value to int if it's a string
        if isinstance(true_value, str):
            try:
                true_value = int(true_value)
            except ValueError:
                return jsonify({"error": "true_value must be a number"}), 400
        
        # Update the record in the database
        try:
            p = Prediction.get(Prediction.observation_id == observation_id)
            p.true_value = true_value
            p.save()
        except Prediction.DoesNotExist:
            # If record doesn't exist, create a new one
            p = Prediction.create(
                observation_id=observation_id,
                true_value=true_value,
                prediction=None,
                observation="{}"
            )
            p.save()
        
        # Return exactly the format specified
        return jsonify({
            "observation_id": observation_id,
            "true_value": str(true_value)
        }), 200
        
    except Exception as e:
        # Handle unexpected errors
        print(f"Error in update: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/list-db-contents')
def list_db_contents():
    """List all predictions in the database."""
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


@app.route('/health')
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"}), 200


# End webserver stuff
########################################

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', debug=False, port=port)