# importing the libraries.

import face_recognition
import cv2
import numpy as np
import pywhatkit
import pafy

# url of the video
url = "https://www.youtube.com/watch?v=S2Oh4cqEmOg"

# # creating pafy object of the video
video = pafy.new(url)

# getting best stream
best = video.getbest(preftype="any")

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(best.url)


# Check if video opened successfully
if cap.isOpened() == False:
    print("Error opening video  file")



# Loading sample pictures and learn how to recognize them.

mussadiq_image = face_recognition.load_image_file("known/mussadiq-malik.jpg")
mussadiq_face_encoding = face_recognition.face_encodings(mussadiq_image)[0]

imran_image = face_recognition.load_image_file("known/imran-khan.jpg")
imran_face_encoding = face_recognition.face_encodings(imran_image)[0]


kashif_image = face_recognition.load_image_file("known/kashif-Jabbar.jpg")
kashif_face_encoding = face_recognition.face_encodings(kashif_image)[0]

sheikh_image = face_recognition.load_image_file("known/sheikh-rasheed.jpg")
sheikh_face_encoding = face_recognition.face_encodings(sheikh_image)[0]

bilawal_image = face_recognition.load_image_file("known/bilawal-bhutto.jpg")
bilawal_face_encoding = face_recognition.face_encodings(bilawal_image)[0]


muaradAli_image = face_recognition.load_image_file("known/murad-ali-shah.jpg")
muradAli_face_encoding = face_recognition.face_encodings(muaradAli_image)[0]

zardari_image = face_recognition.load_image_file("known/asif-zardari.jpg")
zardari_face_encoding = face_recognition.face_encodings(zardari_image)[0]

maira_khan_image = face_recognition.load_image_file("known/maira-khan.jpg")
maira_khan_face_encoding = face_recognition.face_encodings(maira_khan_image)[0]

humayun_saeed_image = face_recognition.load_image_file("known/humayun-saeed.jpg")
humayun_saeed_face_encoding = face_recognition.face_encodings(humayun_saeed_image)[0]

arif_alvi_image = face_recognition.load_image_file("known/arif-alvi.jpg")
arif_alvi_face_encoding = face_recognition.face_encodings(arif_alvi_image)[0]

shahbaz_sharif__image = face_recognition.load_image_file("known/shahbaz-sharif.jpg")
shahbaz_sahrif_face_encoding = face_recognition.face_encodings(shahbaz_sharif__image)[0]

fahad_mustafa_image = face_recognition.load_image_file("known/fahad-mustafa.jpg")
fahad_mustafa_face_encoding = face_recognition.face_encodings(fahad_mustafa_image)[0]

fawad_khan_image = face_recognition.load_image_file("known/fawad-khan.jpg")
fawad_khan_face_encoding = face_recognition.face_encodings(fawad_khan_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    imran_face_encoding,
    kashif_face_encoding,
    mussadiq_face_encoding,
    sheikh_face_encoding,
    bilawal_face_encoding,
    muradAli_face_encoding,
    zardari_face_encoding,
    maira_khan_face_encoding,
    humayun_saeed_face_encoding,
    arif_alvi_face_encoding,
    shahbaz_sahrif_face_encoding,
    fahad_mustafa_face_encoding,
    fawad_khan_face_encoding,

]
known_face_names = [
    "imran khan",
    "kashif Jabbar",
    "Mussadiq Malik",
    "sheikh Rasheed",
    "Bilawal Bhutto",
    "Murad Ali Shah",
    "Asif Ali Zardari",
    "Maira Khan",
    "Humayun Saeed",
    "Arif Alvi",
    "Shahbaz Sharif",
    "Fahad Mustafa",
    "Fawad khan",
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    # cv2.namedWindow(cv2.WINDOW_AUTOSIZE)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []

        #Setting the tolerance value to 0.55 for strict process.
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, 0.55
            )
            name = "Unknown"

           
            # use the known face with the smallest distance to the new face

            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                # pywhatkit.sendwhatmsg_to_group_instantly("KjdmxByr94u4966DodwKim", "Hey *{}*  was just seen!".format(name),tab_close=True)

            face_names.append(name)
            if name in known_face_names:
                print(name)
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
        cv2.rectangle(
            frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow("Real time Face recognition from YouTube Stream", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
