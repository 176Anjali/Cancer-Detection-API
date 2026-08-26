import os

# Allowed image extensions
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


def is_allowed_file(filename: str) -> bool:
    """
    Check whether the uploaded file has a supported image extension.
    """
    extension = os.path.splitext(filename)[1].lower()
    return extension in ALLOWED_EXTENSIONS