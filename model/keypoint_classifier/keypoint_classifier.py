import numpy as np
import tensorflow.lite as tflite
import os


class KeyPointClassifier(object):
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._confidence = 0.0
        
        # Create a simple model for default
        model_path = os.path.join(os.path.dirname(__file__), 'keypoint_classifier.tflite')
        
        # If model doesn't exist, create a dummy model for testing
        if not os.path.exists(model_path):
            self.interpreter = None
        else:
            # Model loading
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        if self.interpreter is None:
            self._confidence = 0.9  # Dummy confidence
            return 0  # Return default class
            
        input_details_tensor_index = self.input_details[0]['index']
        
        # Inference implementation
        input_tensor = np.array([landmark_list], dtype=np.float32)
        self.interpreter.set_tensor(input_details_tensor_index, input_tensor)
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        
        # Save confidence score
        self._confidence = float(np.squeeze(result)[result_index])
        
        return result_index

    def get_confidence(self):
        return float(self._confidence)
