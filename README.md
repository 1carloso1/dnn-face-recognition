# Face Recognition using Deep Neural Networks with OpenCV

This project implements a facial recognition system using deep neural networks (DNN) with OpenCV. The pretrained model is loaded through the Caffe architecture and used to detect faces in images, videos, or in real-time via a webcam.

## Setup Instructions

### Step 1: Activate the Virtual Environment

To get started, activate the virtual environment where all dependencies are already installed. Run the following command:

```bash
source env/bin/activate  # On Windows use 'env\Scripts\activate'
```
###  Step 2: Run the Script

Once the virtual environment is activated, run the main.py script:

```bash
python main.py
```

### Step 3: Choose an Option

After running the script, you will be prompted with the following options:

```bash
-----WELCOME TO THE FACIAL DETECTOR SYSTEM-----
-What kind of file do you wanna try?=-
-1. Image
-2. Video
-3. Webcam
-0. LEAVE
```

Select one of the following options:

- Image: Load an image for face detection.
- Video: Load a video to detect faces in each frame.
- Webcam: Use the webcam for real-time face detection.
- Leave: Exit the system.

The script will automatically detect and process all files in the pruebas_multimedia folder, which contains the images and/or videos for facial detection.

### Preloaded Files

The necessary model files (deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel) are already included in the repository under the pruebas_multimedia folder, so you don't need to manually download them.

This system is ideal for security, monitoring, and real-time image analysis applications.