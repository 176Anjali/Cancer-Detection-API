# -------------------------------------------------------
# Prediction Labels
# -------------------------------------------------------

PREDICTION_LABELS = {

    0: {
        "class": "Benign",
        "message": "The uploaded blood smear image is classified as Benign. No leukemia cells were detected by the AI model."
    },

    1: {
        "class": "Early",
        "message": "The uploaded blood smear image is classified as Early-stage Acute Lymphoblastic Leukemia (ALL). Clinical evaluation is recommended."
    },

    2: {
        "class": "Pre",
        "message": "The uploaded blood smear image is classified as Pre-B Acute Lymphoblastic Leukemia (ALL). Please consult a qualified hematologist for further diagnosis."
    },

    3: {
        "class": "Pro",
        "message": "The uploaded blood smear image is classified as Pro-B Acute Lymphoblastic Leukemia (ALL). Immediate medical evaluation is recommended."
    }

}


def get_prediction(index):
    """
    Returns prediction details for the predicted class.
    """

    return PREDICTION_LABELS.get(
        index,
        {
            "class": "Unknown",
            "message": "The AI model could not confidently classify the uploaded image."
        }
    )