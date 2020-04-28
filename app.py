import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import face_recognition
import cv2
import glob

RECOGNITION_TOLERANCE = 0.57

class Person:
    def __init__(self, name, encoding):
        self.name = name
        self.encoding = encoding

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
image_files = glob.glob("known_persons/*.png")
known_persons = []
for image in image_files:
    loaded_image = face_recognition.load_image_file(image)
    person_name = image.split("/")[1].split(".")[0]
    # append encoding of the first face in file (assuming only 1 face per file)
    loaded_encoding = face_recognition.face_encodings(loaded_image)[0];
    known_persons.append(Person(person_name, loaded_encoding))

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame, 1, 'hog')
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Get list of values how close the face is to the trained models (smaller is better)
            distances = face_recognition.face_distance([p.encoding for p in known_persons], face_encoding)
            closest_match = min(zip([p.name for p in known_persons], distances), key=lambda x: x[1])
            if closest_match[1] < RECOGNITION_TOLERANCE:
                face_names.append(closest_match[0])

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
