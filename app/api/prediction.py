from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
import uuid

from app.services.predictor_service import predict
from app.utils.file_utils import is_allowed_file

# -------------------------------------------------------
# Router
# -------------------------------------------------------

router = APIRouter()

# -------------------------------------------------------
# Upload Folder
# -------------------------------------------------------

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------------

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    # Check MIME type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed."
        )

    # Check file extension
    if not is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Supported formats: jpg, jpeg, png, bmp."
        )

    # Create unique filename
    extension = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{extension}"

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    # Save uploaded image
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Predict
        result = predict(filepath)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        # Delete uploaded image
        if os.path.exists(filepath):
            os.remove(filepath)