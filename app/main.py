from fastapi import FastAPI
from app.api.prediction import router as prediction_router

app = FastAPI(
    title="Cancer Detection API",
    description="AI-powered Cancer Detection using Deep Learning",
    version="1.0.0"
)

# Include API Routes
app.include_router(
    prediction_router,
    prefix="/api",
    tags=["Prediction"]
)


@app.get("/")
def root():
    return {
        "message": "Cancer Detection API is Running Successfully",
        "version": "1.0.0",
        "status": "Healthy"
    }