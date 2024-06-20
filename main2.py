from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from firebase_admin import credentials, auth, initialize_app, firestore
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import logging
import httpx
import os

# Initialize FastAPI app
app = FastAPI()

# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")
initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

# Initialize the Keras model
model = tf.keras.models.load_model('model.h5')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Models
class User(BaseModel):
    email: str
    password: str

# Utility functions
def fetch_fields(doc):
    return {
        'place_id': doc.id,
        'name': doc.get('name'),
        'rating': doc.get('rating'),
        'reviews_count': doc.get('reviews_count'),
        'address': doc.get('address'),
        'lat': doc.get('lat'),
        'long': doc.get('long'),
        'category': doc.get('category'),
        'image_url': f"http://34.101.153.83:3000/img/{doc.id}.jpg",
        'caption_idn': doc.get('caption_idn'),
        'caption_eng': doc.get('caption_eng')
    }

def fetch_lon_lat(doc):
    return {
        'place_id': doc.id,
        'name': doc.get('name'),
        'lat': doc.get('lat'),
        'long': doc.get('long')
    }

# Mount the static files directory
app.mount("/img", StaticFiles(directory="img"), name="img")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "REST API for Journey on Solo"}

# Endpoint Sign-Up
@app.post("/signup")
async def signup(user: User):
    try:
        user_record = auth.create_user(
            email=user.email,
            password=user.password,
            email_verified=False,
            disabled=False
        )
        return user_record
    except Exception as e:
        raise HTTPException(status_code=400, detail={"message": "Signup failed", "error": str(e)})

# Endpoint Login
@app.post("/login")
async def login(user: User):
    try:
        api_key = "AIzaSyDxCpXsq-ZSEVHKXqZgO-059JaALp9mXWY"  # Replace with your API Key
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={
                "email": user.email,
                "password": user.password,
                "returnSecureToken": True
            })

        if response.status_code != 200:
            error_message = response.json().get("error", {}).get("message", "Unknown error")
            raise HTTPException(status_code=400, detail={"message": "Login failed", "error": error_message})

        id_token = response.json().get("idToken")

        return {"message": "Login successful", "token": id_token}
    except Exception as e:
        raise HTTPException(status_code=400, detail={"message": "Login failed", "error": str(e)})

# Firestore data endpoints
@app.get("/data")
async def get_data():
    try:
        collection_ref = db.collection('location')
        docs = collection_ref.stream()
        formatted_data = [fetch_fields(doc) for doc in docs]
        return formatted_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/{place_id}")
async def get_detail_data(place_id: str):
    try:
        collection_ref = db.collection('location')
        query_ref = collection_ref.where('place_id', '==', place_id).limit(1).get()
        for doc in query_ref:
            return fetch_fields(doc)
        raise HTTPException(status_code=404, detail=f"Document with place_id '{place_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/coordinates")
async def get_all_coordinates():
    try:
        collection_ref = db.collection('location')
        docs = collection_ref.stream()
        coordinates = [fetch_lon_lat(doc) for doc in docs]
        return coordinates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/coordinates/{place_id}")
async def get_coordinates(place_id: str):
    try:
        collection_ref = db.collection('location')
        query_ref = collection_ref.where('place_id', '==', place_id).limit(1).get()
        for doc in query_ref:
            return fetch_lon_lat(doc)
        raise HTTPException(status_code=404, detail=f"Document with place_id '{place_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        
        if 'features' not in data:
            raise HTTPException(status_code=400, detail={"error": "'features' key not found in the request data"})
        
        features = data['features']
        
        # Ensure features is a list of floats
        if not isinstance(features, list):
            raise HTTPException(status_code=400, detail={"error": "'features' must be a list"})
        
        # Convert features to a numpy array and reshape it to match the model input
        input_data = np.array([features])
        logging.debug(f"Input data shape: {input_data.shape}")
        
        # Predict using the model
        predictions = model.predict(input_data)
        logging.debug(f"Model predictions: {predictions}")
        
        # Extract the predicted class or return the raw predictions based on your requirement
        predicted_class = np.argmax(predictions, axis=1).tolist()
        logging.debug(f"Predicted class: {predicted_class}")
        
        return {"predictions": predicted_class}
    except ValueError as ve:
        logging.error(f"Value error: {str(ve)}")
        raise HTTPException(status_code=400, detail={"error": f"Invalid input: {str(ve)}"})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": f"Prediction error: {str(e)}"})

# Start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
