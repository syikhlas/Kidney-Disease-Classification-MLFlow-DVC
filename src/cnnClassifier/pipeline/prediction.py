import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load the trained model
        model_path = os.path.join("model", "model.h5")
        model = load_model(model_path)

        # Preprocess the input image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0  # Normalize to [0, 1]
        test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

        # Check if image is valid or blank
        if test_image.std() < 0.01:
            return [{"error": "Image appears to be blank or invalid."}]

        # Make prediction
        prediction = model.predict(test_image)
        confidence = float(prediction[0][0])  # Tumor probability

        # Define threshold for classification
        threshold = 0.5  # Standard threshold for binary classification

        # Classification logic
        if 0.3 <= confidence <= 0.7:
            label = "Uncertain or Invalid Image"
        elif confidence > threshold:
            label = "Tumor"
        else:
            label = "Normal"

        return [{"image": label, "confidence": confidence}]