import cv2
import numpy as np

labels = ['0x0', '1x0', '1x1', '2x0', '2x1', '2x2', '3x0', '3x1', '3x2', '3x3', '4x0', '4x1', '4x2', '4x3', '4x4', '5x0', '5x1', '5x2', '5x3', '5x4', '5x5', '6x0', '6x1', '6x2', '6x3', '6x4', '6x5', '6x6']
number = [0, 1, 2, 2, 3, 4, 3, 4, 5, 6, 4, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 11, 12]

def preprocess_image(file) -> np.ndarray:
    '''
    Preprocess the input image to be similar with the training data.
    '''
    # Read image
    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_GRAYSCALE)

    # Resize image
    img = cv2.resize(img, (100, 100))

    # Normalize image
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img
