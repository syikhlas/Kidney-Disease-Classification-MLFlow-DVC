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

        # Make predictions
        predictions = model.predict(test_image)
        confidence = float(np.max(predictions))  # Highest confidence score
        result = int(np.argmax(predictions, axis=1)[0])  # Predicted class index

        # Define confidence threshold
        threshold = 0.80 # Adjust this threshold based on your model
        
        

        # Classification based on confidence
        if confidence >= threshold:
            if result == 1:
                prediction = "Tumor"
            else:
                prediction = "Normal"
            return [{"image": prediction, "confidence": confidence}]
        else:
            return [{"error": "Invalid input, please specify a valid image."}]
