import os
import io
import json
import pandas as pd
from quart import Quart, request, Response, send_file
from model import Model
from request import ModelRequest

app = Quart(__name__)

# Initialize the model to be used for inference.
model = None

@app.before_serving
async def startup():
    """This function is called once before the server starts to initialize the model."""
    global model
    model = Model(app)

@app.route('/embed', methods=['POST'])
async def embed():
    """This endpoint receives a CSV file, extracts text data from it, and uses the model to generate embeddings and topic information."""
    global model

    files = await request.files  # Get the uploaded files
    uploaded_file = files.get('file')  # Get the uploaded CSV file

    if not uploaded_file:
        return Response(json.dumps({"error": "No file uploaded"}), status=400, mimetype='application/json')

    # Read the CSV file into a DataFrame
    csv_data = pd.read_csv(io.BytesIO(uploaded_file.stream.read()))

    # Extract the text data
    text_data = csv_data['text'].tolist()

    # Create a ModelRequest object with the extracted text data
    req = ModelRequest(text=text_data)

    # Call the model's inference method and get the response
    response = await model.inference(req)

    if response is None:
        # If an error occurred during inference, return an error response
        return Response(json.dumps({"error": "Inference error"}), status=500, mimetype='application/json')

    # Convert the CSV string from the response into a DataFrame
    df = pd.read_csv(io.StringIO(response))

    # Save the DataFrame to a CSV file
    output_file_path = 'output.csv'
    df.to_csv(output_file_path, index=False)

    # Send the CSV file back as a download response
    return await send_file(output_file_path, mimetype='text/csv', as_attachment=True, attachment_filename='output.csv')