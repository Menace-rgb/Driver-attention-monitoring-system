from DetectEye import EyeDetector
import cv2
import mediapipe as mp
import numpy as np
from Score_Evaluation import Score_Evaluation
from HeadOrientation import HeadOrientation

def process_face_mediapipe(frame):

    results = faceMesh.process(frame)
    return results.multi_face_landmarks



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
score_evaluation = Score_Evaluation(11, ear_tresh=0.15, ear_time_tresh=2, gaze_tresh=0.3,
                       gaze_time_tresh=2, pitch_tresh=35, yaw_tresh=28, pose_time_tresh=2.5, verbose=False)
if camera_matrix is not None and dist_coeffs is not None:
    headOrientation = HeadOrientation(camera_matrix= camera_matrix,dist_coeffs=dist_coeffs,show_axis=True)
else:
    headOrientation = HeadOrientation(show_axis= True)
cap = cv2.VideoCapture('./yawning.mp4')

# window_width = 800
window_width = 350
frame_counter = 0

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 1)
avg_gaze_score = None





while(cap.isOpened()):
    ret, img = cap.read()
    if ret:
        new_frame = np.zeros((500, 500, 3), np.uint8)
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

        grayscale = cv2.bilateralFilter(grayscale, 5, 10, 10)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        multi_face_landmarks = process_face_mediapipe(imgRGB)
        eye_detector.landmark_display_and_coordinates_noted(img, multi_face_landmarks, width , height)
        eye_detector.RegionOfInterest_display_and_tracked(grayscale, img)

        gaze_eye_left, left_eye =eye_detector.Gaze_calculation(eye_detector.region_of_interest_left)
        # eye_detector.Gaze_calculation(eye_detector.region_of_interest_left)
        gaze_eye_right, right_eye = eye_detector.Gaze_calculation(eye_detector.region_of_interest_right)

        right_eye_pupil = (eye_detector.positionEstimator(eye_detector.region_of_interest_right))

        left_eye_pupil = (eye_detector.positionEstimator(eye_detector.region_of_interest_left))

        if gaze_eye_left and gaze_eye_right:

            # computes the average gaze score for the 2 eyes
            avg_gaze_score = (gaze_eye_left + gaze_eye_right) / 2
        elif gaze_eye_left:
            avg_gaze_score = gaze_eye_left
        else:
            avg_gaze_score = gaze_eye_right

        EAR = eye_detector.get_EAR()

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

        left_gaze = eye_detector.gaze_another_method(eye_detector.region_of_interest_left)
        right_gaze = eye_detector.gaze_another_method(eye_detector.region_of_interest_right)
        
        gaze_score = (left_gaze + right_gaze) / 2

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        if EAR is not None:
            cv2.putText(img, "EAR:" + str(round(EAR, 3)), (10, 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

        # if avg_gaze_score is not None:
        #     cv2.putText(img, "Gaze Score:" + str(round(avg_gaze_score, 3)), (10, 80),
        #                         cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

        if tired:
            cv2.putText(img, "TIRED!", (10, 280),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "FRESH!", (10, 280),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)


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
            cv2.putText(img, "ASLEEP!", (10, 400),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2, cv2.LINE_AA)
        if looking_away:
            cv2.putText(img, "LOOKING AWAY!", (10, 350),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2, cv2.LINE_AA)
        if distracted:
            cv2.putText(img, "DISTRACTED!", (10, 400),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2, cv2.LINE_AA)
            
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


