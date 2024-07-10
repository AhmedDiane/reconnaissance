from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import face_recognition
import os

app = FastAPI()

# Charger les signatures au démarrage de l'application
try:
    signatures_class = np.load('faceSignatures_db.npy')
    X = signatures_class[:, :-1].astype('float')
    Y = signatures_class[:, -1]
    print(f"Loaded {X.shape[0]} face signatures.")
except Exception as e:
    print(f"Erreur lors du chargement des signatures : {e}")
    X = None
    Y = None

@app.post("/recognize-face/")
async def recognize_face(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    imgR = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
    facesCurrent = face_recognition.face_locations(imgR)
    encodesCurrent = face_recognition.face_encodings(imgR, facesCurrent)

    results = []
    for encodeFace, faceLoc in zip(encodesCurrent, facesCurrent):
        matches = face_recognition.compare_faces(X, encodeFace)
        faceDis = face_recognition.face_distance(X, encodeFace)
        matchIndex = np.argmin(faceDis)
        name = "Unknown"
        if matches[matchIndex]:
            name = Y[matchIndex].upper()
        results.append({
            "name": name,
            "location": faceLoc,
            "distance": faceDis[matchIndex]
        })

    return {"results": results}
