# Face Detection and Matching
Manar Vink (24025838), Anna Tan (24214464)

## Setup

Install dependencies with:
```
pip install pillow numpy opencv-contrib-python mtcnn tensorflow
```

Must use `opencv-contrib-python`**, not `opencv-python` - the contrib version includes the SFace recognition model. If `opencv-python` is already installed:
```
pip uninstall opencv-python
pip install opencv-contrib-python
```

PIL, numpy, cv2, and mtcnn must be installed before the program can be run.

## Required files

Both files must be in the same folder:
- `CITS4402_CVproject.py`
- `face_recognition_sface_2021dec.onnx` - submitted with this project, but can also be downloaded from the [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) GitHub repository under `models/face_recognition_sface/`

## Run

```
python CITS4402_CVproject.py
```
