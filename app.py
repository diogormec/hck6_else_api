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

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')


# Adicione logo após:
try:
    DB.execute_sql('ALTER TABLE prediction ADD COLUMN prediction INTEGER;')
except Exception as e:
    print(f"Column may already exist: {e}")


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

@app.before_request
def before_request():
    """Connect to the database before each request."""
    DB.connect()

@app.after_request
def after_request(response):
    """Close the database connection after each request."""
    DB.close()
    return response

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
        
        # Filter to only include columns expected by the model
        model_data = {}
        for col in columns:
            if col in data:
                model_data[col] = data[col]
            else:
                # Handle missing columns by setting them to None (will be imputed)
                model_data[col] = None
        
        # Convert to DataFrame
        obs = pd.DataFrame([model_data])
        
        # Convert data types according to the model expectations
        for col, dtype in dtypes.items():
            if col in obs.columns:
                try:
                    if dtype == 'object':
                        obs[col] = obs[col].astype(str)
                    else:
                        obs[col] = pd.to_numeric(obs[col], errors='coerce')
                except Exception as e:
                    print(f"Warning: Could not convert {col} to {dtype}: {e}")
        
        # Generate the prediction (round to nearest integer for length of stay)
        prediction = int(round(pipeline.predict(obs)[0]))
        
        # Format the response exactly as specified
        response = {
            "observation_id": observation_id,
            "prediction": str(prediction)  # Convert to string as per requirements
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
    try:
        obs = request.get_json()
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if not obs or 'observation_id' not in obs or 'true_value' not in obs:
        return jsonify({"error": "'observation_id' and 'true_value' are required"}), 400
    if not obs['true_value']:
        return jsonify({"error": "'true_value' can't be empty."}), 400
    try:
        observation_id = obs['observation_id']
        true_value = int(obs['true_value']) if isinstance(obs['true_value'], str) else obs['true_value']
        
        try:
            p = Prediction.get(Prediction.observation_id == observation_id)
            p.true_value = true_value
            p.save()
        except Prediction.DoesNotExist:
            # Não permitir criação de novos registros sem prediction
            return jsonify({
                "error": "Observation ID not found. Please create a prediction first.",
                "observation_id": observation_id
            }), 404
        
        return jsonify({
            "observation_id": observation_id,
            "true_value": str(true_value)
        }), 200
        
    except ValueError:
        return jsonify({"error": "true_value must be a number"}), 400
    except Exception as e:
        print(f"Error in update: {e}")
        return jsonify({"error": str(e)}), 500


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