import sys, os, cv2, time, csv, numpy as np
import math
import dlib
import glob
from skimage import io
from scipy.spatial import distance
import DataBaseJson as dbjsn
import models

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def main():
    if len(sys.argv) != 3:
        print("insert please python3 DVAN.py CameraNumber(0 or 1) and accuracy (m for minutes, h for hours and w for weekdays)")
        exit(1)

    face_rec_model_path = './models/dlib_face_recognition_resnet_model_v1.dat'
    predictor_path = './models/shape_predictor_68_face_landmarks.dat'

    numcam = int(sys.argv[1])
    acc = sys.argv[2]

    AGS_model = models.WideResNetCreater()()
    img_size = 64

    AGS_model.load_weights(os.path.join("models/", "weights.18-4.06.hdf5"))


    dbfile = 'camera{numcam}_db.json'.format(numcam=numcam)
    detector = dlib.get_frontal_face_detector() # load face detection model
    sp = dlib.shape_predictor(predictor_path) # load shape predictor model for face recognition
    facerec = dlib.face_recognition_model_v1(face_rec_model_path) # load facre recogniton model
    cap = cv2.VideoCapture(numcam)

    RefNumFace = 0#предыдущий
    TargNumFace = 0#текущий


    while True:
        ret, frame = cap.read()
        img_h, img_w, img_ch = np.shape(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector(gray)
        TargNumFace = len(dets)
        if TargNumFace > RefNumFace:
            dbjsn.faceRecogn(shape_predictor=sp,deltha = 0.03, recognizer=facerec, coordinates=dets,acc = acc, gray=gray, frame=frame,AGSmodel=AGS_model, filename=dbfile, img_size = img_size) # face recognition
        cv2.imshow('camera '+str(numcam), frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        RefNumFace = TargNumFace
    cv2.destroyAllWindows()
    cap.release()

main()
