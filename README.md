AMERICAN SIGN LANGUAGE (ASL) DETECTION


PROJECT OVERVIEW
This project implements a deep learning–based system to recognize
American Sign Language (ASL) hand gestures from images. A Convolutional
Neural Network (CNN) is used to automatically learn visual features
from hand gesture images and classify them into their corresponding
ASL alphabets or symbols.

The system is designed to be dynamic and scalable, automatically
adapting to the number of gesture classes available in the dataset.


OBJECTIVE
- Detect ASL hand signs from input images
- Classify gestures into alphabets or special symbols
- Build a reusable and scalable CNN-based architecture
- Provide a foundation for real-time gesture recognition systems


DATASET
Source: Kaggle – ASL Alphabet Dataset

Dataset Structure:
- Images are organized into folders by class name
- Each folder represents one ASL gesture
- The current dataset subset contains multiple alphabet classes
- The model dynamically detects the number of classes during training


TECHNOLOGY STACK
- Programming Language: Python
- Deep Learning Framework: TensorFlow, Keras
- Image Processing: OpenCV
- Data Handling: NumPy
- Visualization: Matplotlib
- Environment: Anaconda


SYSTEM ARCHITECTURE
The system follows a modular deep learning architecture:

1. ASL Image Input
   - Static image containing a hand gesture

2. Image Preprocessing
   - Resize to 64×64
   - Pixel normalization

3. CNN Feature Extraction
   - Convolutional layers
   - MaxPooling layers

4. Fully Connected Layers
   - Dense layers
   - Dropout for regularization

5. Softmax Classification
   - Dynamic class count
   - Outputs predicted ASL label


MODEL DETAILS
- Input Size: 64 × 64 × 3
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 15
- Batch Size: 32


PROJECT STRUCTURE
ASL_Detection/
│
├── dataset/
│   └── ASL_Dataset/
│       └── asl_alphabet_train/
│           └── asl_alphabet_train/
│               ├── A/
│               ├── B/
│               ├── ...
│               └── del/
│
├── model/
│   └── asl_model.h5
│
├── train_asl.py
├── predict_asl.py
└── redem.txt


HOW TO RUN
1. Activate Anaconda environment
   conda activate base

2. Train the model
   python train_asl.py

3. Run prediction
   python predict_asl.py

OUTPUT
- Trained CNN model saved as:
  model/asl_model.h5
- Prediction script outputs:
  - Predicted ASL label
  - Confidence score

APPLICATIONS
- Assistive technology for the hearing-impaired
- Human–computer interaction systems
- Educational tools for learning sign language
- Gesture-based user interfaces

LIMITATIONS
- Works on static images only
- Sensitive to lighting and hand positioning
- Real-time performance not implemented in current version

CONCLUSION
This project demonstrates the practical application of Convolutional
Neural Networks for visual gesture recognition. It highlights how deep
learning can be used to build accessible and intelligent systems for
real-world communication challenges.


AUTHOR
------
American Sign Language Detection Project

