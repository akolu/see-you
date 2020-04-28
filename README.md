# SeeYou
This project is a demonstration how a face recognition library can be used to recognize & track faces from webcam in real time. 

# Requirements
* [dlib](http://dlib.net/) + its requirements (cmake, jpeg, libpng)
* [opencv](http://opencv.org/) + its requirements (cmake, pkg-config, eigen, ffmpeg, jpeg, libpng, libtiff, openexr, numpy)
* [face_recognition](https://github.com/ageitgey/face_recognition), a face recognition api which uses dlib
    
See requirements.txt for more info.
    
# Usage
1. create virtualenv for the project
2. `pip install cmake jpeg libpng pkg-config eigen ffmpeg libtiff openexr numpy`
3. `brew install dlib opencv`
4. `pip install face_recognition`
5. put some images in known_persons folder (must be in .png format)
6. run the app with `python app.py`
    