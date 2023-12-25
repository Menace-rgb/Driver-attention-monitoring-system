from DetectEye import EyeDetector
import cv2
import mediapipe as mp
import numpy as np
from Score_Evaluation import Score_Evaluation
from HeadOrientation import HeadOrientation
from YawnDetection import YawnDetection
import pygame
from pygame import mixer
import time

mixer.init()


yawning_music = mixer.Sound('Voices/yawning.mp3')
looking_left = mixer.Sound('Voices/Looking-Left.mp3')
looking_right = mixer.Sound('Voices/Looking-Right.mp3')
looking_up = mixer.Sound('Voices/Looking-Up.mp3')
looking_down = mixer.Sound('Voices/Looking-Down.mp3')
asleep_music = mixer.Sound('Voices/Asleep.mp3')
forward = mixer.Sound('Voices/Forward.mp3')

asleep_prev = 0
asleep_current = 0
yawn_prev = 0
yawn_current = 0

# mediapipe providing the important face landmarks
def process_face_mediapipe(frame):

    results = faceMesh.process(frame)
    return results.multi_face_landmarks

#

camera_matrix = np.array(
    [[534.07088364,   0.,         341.53407554],
 [  0.,         534.11914595, 232.94565259],
 [  0.,           0. ,          1.        ]], dtype="double")

# distortion coefficients obtained from the camera calibration script, using a 9x6 chessboard
dist_coeffs = np.array(
    [[-2.92971637e-01,  1.07706962e-01,  1.31038376e-03, -3.11018781e-05,
   4.34798110e-02]], dtype="double")


# camera_matrix = None
# dist_coeffs = None
# instantiaiont
eye_detector = EyeDetector(showProcessing= False)
# score_evaluation = Score_Evaluation(capture_fps = 11, EAR_THRESHOLD= 0.15, EAR_TIME_THRESHOLD=2, GAZE_THRESHOLD=0.2, GAZE_TIME_THRESHOLD= 2, PITCH_THRESHOLD=35, YAW_THRESHOLD=28, POSE_TIME_THRESHOLD= 2.5)
score_evaluation = Score_Evaluation(11, ear_tresh=0.25, ear_time_tresh=4.0, gaze_tresh=0.2,
                       gaze_time_tresh=2, pitch_tresh=35, yaw_tresh=28, pose_time_tresh=2.5, verbose=False)

yawn_detection = YawnDetection()

if camera_matrix is not None and dist_coeffs is not None:
    headOrientation = HeadOrientation(looking_left,looking_right,looking_up,looking_down,forward,mixer,camera_matrix= camera_matrix,dist_coeffs=dist_coeffs,show_axis=True)
else:
    headOrientation = HeadOrientation(looking_left,looking_right,looking_up,looking_down,forward,mixer,show_axis= True)
    
cap = cv2.VideoCapture(0)


ret, camera = cap.read()
img_hieght, img_width = camera.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output21.mp4', fourcc, 30.0, (img_width, img_hieght))
# videowriter saving the output of the video in a file


window_width = 800

frame_counter = 0
current_time = 0
previous_time = 0

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 1)
avg_gaze_score = None
yawn_counter = 0
asleep_counter = 0

while(cap.isOpened()):
    yawning = False
    ret, img = cap.read()
    img = cv2.flip(img,1)
    if ret:
        # new_frame = np.zeros((500, 500, 3), np.uint8)
        frame_counter = frame_counter + 1
        h,w,c = img.shape
        # print(img.shape)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print("width is ", width)
        # print("height is ", height)
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resized_Window", (window_width, int((height / width) * window_width)))

        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # converting BGR to grayscale

        grayscale = cv2.bilateralFilter(grayscale, 5, 10, 10)
        # applying a bilatereal filter

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        multi_face_landmarks = process_face_mediapipe(imgRGB)
        eye_detector.landmark_display_and_coordinates_noted(img, multi_face_landmarks, width , height)
        eye_detector.RegionOfInterest_display_and_tracked(grayscale, img)

        gaze_eye_left, left_eye =eye_detector.Gaze_calculation(eye_detector.region_of_interest_left)
        # print(gaze_eye_left)
        # eye_detector.Gaze_calculation(eye_detector.region_of_interest_left)
        gaze_eye_right, right_eye = eye_detector.Gaze_calculation(eye_detector.region_of_interest_right)
        # print(gaze_eye_right)

        # right_eye_pupil = (eye_detector.positionEstimator(eye_detector.region_of_interest_right))

        # left_eye_pupil = (eye_detector.positionEstimator(eye_detector.region_of_interest_left))
        if gaze_eye_left and gaze_eye_right:

            # computes the average gaze score for the 2 eyes
            avg_gaze_score = (gaze_eye_left + gaze_eye_right) / 2
            
        elif gaze_eye_left:
            avg_gaze_score = gaze_eye_left
            # if only one eye available
        else:
            avg_gaze_score = gaze_eye_right

        EAR = eye_detector.get_EAR()
        # print(EAR)

        frame_det, roll, pitch, yaw = headOrientation.get_pose(
                    frame=img, landmarks=multi_face_landmarks, width = width, height = height)

        tired, perclos_score = score_evaluation.get_PERCLOS(EAR)
        
        # if right_eye_pupil:
        #     cv2.putText(img, "Right Eye :" + str(right_eye_pupil), (10, 400),
        #                         cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
            
        # if left_eye_pupil:
        #     cv2.putText(img, "Left Eye :" + str(right_eye_pupil), (10, 350),
        #                         cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)



        # print("The perclos score is ", perclos_score)

        # left_gaze = eye_detector.gaze_another_method(eye_detector.region_of_interest_left)
        # right_gaze = eye_detector.gaze_another_method(eye_detector.region_of_interest_right)

        yawn_detection.get_imp_coordinates(multi_face_landmarks,img, width, height)
        lips_distance = yawn_detection.get_distance()
        if lips_distance and lips_distance > 25:
            yawning = True
            yawn_current = time.time()
            yawn_counter += 1
            if yawn_counter > 30:
                yawn_counter = 0
            # print(" The yawn counter is ", yawn_counter)
            # if pygame.mixer.get_busy() == 0 and yawn_counter == 30:
            if pygame.mixer.get_busy() == 0 and yawn_prev == 0 or yawn_current - yawn_prev > 5:
                yawning_music.play()
                yawn_prev = yawn_current
        # gaze_score = (left_gaze + right_gaze) /2 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        if EAR is not None:
            cv2.putText(img, "EAR:" + str(round(EAR, 3)), (10, 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # if avg_gaze_score is not None:
        #     cv2.putText(img, "Gaze Score:" + str(round(avg_gaze_score, 3)), (10, 80),
        #                         cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

        if yawning:
            cv2.putText(img, f"Yawning : {yawning}", (10,310), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2,cv2.LINE_AA)
        else:
            cv2.putText(img, f"Yawning : {yawning}", (10,310), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2,cv2.LINE_AA)
            yawning_counter = 0
        if tired:
            cv2.putText(img, "Tired!", (10, 270),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "Fresh!", (10, 270),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)


        asleep, looking_away, distracted, right, center, left = score_evaluation.score_evaluate(
            EAR, avg_gaze_score, roll, pitch, yaw)

        if right:
            cv2.putText(img, "Right!", (10, 400),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2, cv2.LINE_AA)
        if left:
            cv2.putText(img, "Left!", (10, 400),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2, cv2.LINE_AA)
        if center:
            cv2.putText(img, "Center!", (10, 400),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2, cv2.LINE_AA)
            # if the state of attention of the driver is not normal, show an alert on screen
        if asleep:

            asleep_current = time.time()
            
            asleep_counter += 1
            if asleep_counter > 20:
                asleep_counter = 0
            # print("Asleep counter is ", asleep_counter)
            cv2.putText(img, "Asleep", (10, 350),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
            # if pygame.mixer.get_busy() == 0 and asleep_counter == 20:
            if pygame.mixer.get_busy() == 0 and (asleep_prev == 0) or (asleep_current - asleep_prev > 5):
                asleep_music.play()
                asleep_prev = asleep_current
        else:
            cv2.putText(img, "Awake", (10, 350),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
            asleep_counter = 0

        # if looking_away:
        #     cv2.putText(img, "Pupil Not In Center", (10, 400),
        #                 cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
        # if distracted:
        #     cv2.putText(img, "Distracted", (10, 400),
        #                 cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2, cv2.LINE_AA)
        # if perclos_score is not None:
            
        #     cv2.putText(img, f"PERCLOS : {round(perclos_score,2)}", (10, 450),
        #                 cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
            
        current_time = time.time()
        try:
            fps = 1 / (current_time - previous_time)
        except:
            fps = None


        previous_time = current_time

        if fps is not None:
            cv2.putText(img, f"FPS : {int(fps)}", (500, 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)


        out.write(img)
        
        
        cv2.imshow("Resized_Window", img)

    else:
        
        # print("The frame counter is ", frame_counter)
        # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            break


cap.release()
cv2.destroyAllWindows()


