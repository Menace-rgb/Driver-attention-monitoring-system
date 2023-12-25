import cv2
import numpy as np
from numpy import linalg as LA
import mediapipe as mp


class EyeDetector:
    '''Objectives:
        1. Calculation of EAR(Eye Aperture Rate)
        2. Calculation of Gaze Score Estimation
           
        Methods:
        displayEyeLandmarks : Shows landmark points in eyes
        calculateEar : Average Eye Aperture Rate calculation
        calculateGaze
            
    '''
    def __init__(self, showProcessing : bool = False):
        '''  
           Parameters Involved:
           show_processing : Shows the frame images during processing
        '''
        self.keypoints = None
        self.frame = None
        self.showProcessing = showProcessing
        self.eyeWidth = None

        self.left_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
        self.right_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
        # self.left_eye_landmarks = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163]
        # self.right_eye_landmarks = [362,398,384,385,386,387,388,466,263,382,381,380,374,373,390,249]
        # self.landmarks_for_EAR_calculation = {'l_l_corner' : 33, 'l_r_corner' : 133, 'l_l_u_middle' : 160, 'l_l_b_middle' : 144,
        #              'l_r_u_middle' : 158 , 'l_r_b_middle' : 153,
        #              'r_l_corner' : 362, 'r_r_corner' : 263, 'r_l_u_middle' : 385, 'r_l_b_middle' : 380,
        #              'r_r_u_middle' : 387 , 'r_r_b_middle' : 373}
        self.landmarks_for_EAR_calculation = {'r_l_corner' : 33, 'r_r_corner' : 133, 'r_l_u_middle' : 160, 'r_l_b_middle' : 144,
                     'r_r_u_middle' : 158 , 'r_r_b_middle' : 153,
                     'l_l_corner' : 362, 'l_r_corner' : 263, 'l_l_u_middle' : 385, 'l_l_b_middle' : 380,
                     'l_r_u_middle' : 387 , 'l_r_b_middle' : 373}
        self.left_eye_imp_coordinates = {}
        self.right_eye_imp_coordinates = {}
        self.region_of_interest_left = []
        self.region_of_interest_right = []

            
    def landmark_display_and_coordinates_noted(self, frame, multi_face_landmarks, width, height):
        '''
        Shows all the landmark of the eyes and also notes down all the required coordinates of the landmarks for the EAR calculation
        '''
        # print(multi_face_landmarks)
        if multi_face_landmarks:
            for faceLms in multi_face_landmarks:
                for id, lm in enumerate(faceLms.landmark):
                    if id in self.left_eye_landmarks:
                        req_x, req_y  = int(lm.x * width), int(lm.y * height)
                        cv2.circle(frame, (req_x, req_y), 1, (255, 0, 255), cv2.FILLED)
            

                        if id in self.landmarks_for_EAR_calculation.values():
                            self.left_eye_imp_coordinates[id] = [req_x, req_y]

                    if id in self.right_eye_landmarks:
                        req_x, req_y = int(lm.x * width), int(lm.y * height)
                        cv2.circle(frame,(req_x, req_y),1, (255, 0, 255), cv2.FILLED)

                        if id in self.landmarks_for_EAR_calculation.values():
                            self.right_eye_imp_coordinates[id] = [req_x, req_y]

                # self.RegionOfInterest_display_and_tracked(frame)
    

    def RegionOfInterest_display_and_tracked(self,grayscale, frame):

        '''Displays and Notes down the region of interest of eye in order to track the pupil of eye'''
        # print(self.left_eye_imp_coordinates)

        try:
            max_x_left = max(self.left_eye_imp_coordinates.items(), key = lambda x : x[1][0])
            max_y_left = max(self.left_eye_imp_coordinates.items(), key = lambda x : x[1][1])
            min_x_left = min(self.left_eye_imp_coordinates.items(), key = lambda x : x[1][0])
            min_y_left = min(self.left_eye_imp_coordinates.items(), key = lambda x : x[1][1])
            cv2.rectangle(frame,( min_x_left[1][0] - 2, min_y_left[1][1] - 2),(max_x_left[1][0] + 2, max_y_left[1][1] + 2),(0, 0, 0), 1 )

            # print("Max x left ", max_x_left[1][0])
            # print("Max y left", max_y_left[1][1])
            # print("Min x left", min_x_left[1][0])
            # print("Min y left", min_y_left[1][1])
            # self.region_of_interest_left = [[min_x_left[1][0], min_y_left[1][1]],[max_x_left[1][0], max_y_left[1][1]]]
            # print("The frame is ", frame.shape)

            # self.region_of_interest_left = grayscale[min_y_left[1][1]-2:max_y_left[1][1]+2, min_x_left[1][0]-2:max_x_left[1][0]+2]
            
            self.region_of_interest_left = grayscale[min_y_left[1][1]:max_y_left[1][1], min_x_left[1][0]:max_x_left[1][0]]

            max_x_right = max(self.right_eye_imp_coordinates.items(), key = lambda x : x[1][0])
            max_y_right = max(self.right_eye_imp_coordinates.items(), key = lambda x : x[1][1])
            min_x_right = min(self.right_eye_imp_coordinates.items(), key = lambda x : x[1][0])
            min_y_right = min(self.right_eye_imp_coordinates.items(), key = lambda x : x[1][1])

            cv2.rectangle(frame,( min_x_right[1][0] - 2, min_y_right[1][1] - 2),(max_x_right[1][0] + 2, max_y_right[1][1] + 2),(0, 0, 0), 1 )

            # self.region_of_interest_right = [[min_x_right[1][0], min_y_right[1][1]],[max_x_left[1][0], max_y_right[1][1]]]

            # self.region_of_interest_right = grayscale[min_y_right[1][1]-2:max_y_right[1][1]+2, min_x_right[1][0]-2:max_x_right[1][0]+2]

            self.region_of_interest_right = grayscale[min_y_right[1][1]:max_y_right[1][1], min_x_right[1][0]:max_x_right[1][0]]

            # self.Gaze_calculation()
            # self.combined_region_of_interest = 
            # print("Region of interest left is ", self.region_of_interest_left)
        except:
            pass

    
    def Gaze_calculation(self, region_of_interest):
        # print("The region of interest is ", region_of_interest)

        try:
            eye_center = np.array(
                    [(region_of_interest.shape[1] // 2), (region_of_interest.shape[0] // 2)])  # eye ROI
            
            # print("The eye center is ", eye_center)
            gaze_score = None
            circles = None

                # a bilateral filter is applied for reducing noise and keeping eye details
            # region_of_interest = cv2.bilateralFilter(region_of_interest, 4, 40, 40)
            # print("The region of interest is ", len(region_of_interest))
            circles = cv2.HoughCircles(region_of_interest, cv2.HOUGH_GRADIENT, 1, 10,
                                        param1=90, param2=6, minRadius=1, maxRadius=100)
            # # a Hough Transform is used to find the iris circle and his center (the pupil) on the grayscale region_of_interest image with the contours drawn in white
            # print("Hough Transform detected circle" , circles)
            if circles is not None and len(circles) > 0:
                circles = np.uint16(np.around(circles))
                circle = circles[0][0, :]

                cv2.circle(
                    region_of_interest, (circle[0], circle[1]), circle[2], (0, 0, 0), 1)
                cv2.circle(
                    region_of_interest, (circle[0], circle[1]), 1, (0, 255, 0), -1)

                # pupil position is the first circle center found with the Hough Transform
                pupil_position = np.array([int(circle[0]), int(circle[1])])

                cv2.line(region_of_interest, (eye_center[0], eye_center[1]), (
                    pupil_position[0], pupil_position[1]), (255, 255, 255), 1)

                gaze_score = LA.norm(
                    pupil_position - eye_center) / eye_center[0]
                # computes the L2 distance between the eye_center and the pupil position

            cv2.circle(region_of_interest, (eye_center[0],
                                    eye_center[1]), 1, (0, 0, 0), -1)

            if gaze_score is not None:
                return gaze_score, region_of_interest
            else:
                return None, None
        except:
            return None, None
            


    
    def EAR_calculation(self,eye_pts, direction):
        try:
            if direction == "left":
                a = LA.norm(np.array(eye_pts[self.landmarks_for_EAR_calculation['l_l_u_middle']]) - np.array(eye_pts[self.landmarks_for_EAR_calculation['l_l_b_middle']]))
                b = LA.norm(np.array(eye_pts[self.landmarks_for_EAR_calculation['l_r_u_middle']]) - np.array(eye_pts[self.landmarks_for_EAR_calculation['l_r_b_middle']]))
                c = 2*LA.norm(np.array(eye_pts[self.landmarks_for_EAR_calculation['l_l_corner']]) - np.array(eye_pts[self.landmarks_for_EAR_calculation['l_r_corner']]))
        
            else:
                a = LA.norm(np.array(eye_pts[self.landmarks_for_EAR_calculation['r_l_u_middle']]) - np.array(eye_pts[self.landmarks_for_EAR_calculation['r_l_b_middle']]))
                b = LA.norm(np.array(eye_pts[self.landmarks_for_EAR_calculation['r_r_u_middle']]) - np.array(eye_pts[self.landmarks_for_EAR_calculation['r_r_b_middle']]))
                c = 2*LA.norm(np.array(eye_pts[self.landmarks_for_EAR_calculation['r_l_corner']]) - np.array(eye_pts[self.landmarks_for_EAR_calculation['r_r_corner']]))
            EAR = (a + b) / c
        
            return EAR      
        except:
            return None 
        
        

    def get_EAR(self):
        try:
            left_EAR = self.EAR_calculation(self.left_eye_imp_coordinates, "left")
            right_EAR = self.EAR_calculation(self.right_eye_imp_coordinates, "right")
            avg_EAR = (left_EAR + right_EAR ) / 2
            return avg_EAR
        except:
            return None
    
    def gaze_another_method(self,grayscale_region_of_interest):
        
        _, threshold_eye = cv2.threshold(grayscale_region_of_interest, 70, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Threshold_eye", threshold_eye)
        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)
        if left_side_white == 0:
            gaze_ratio = 1
        elif right_side_white == 0:
            gaze_ratio = 5
        else:
            gaze_ratio = left_side_white / right_side_white
        return gaze_ratio

    def positionEstimator(self,cropped_eye):
        # print("Cropped_eye is ", len(cropped_eye))
        # getting height and width of eye 
        h, w =cropped_eye.shape

        # remove the noise from images
        gaussain_blur = cv2.GaussianBlur(cropped_eye, (9,9),0)
        median_blur = cv2.medianBlur(gaussain_blur, 3)

        # applying thrsholding to convert binary_image
        ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)

        # create fixd part for eye with 
        piece = int(w/3) 

        # slicing the eyes into three parts 
        right_piece = threshed_eye[0:h, 0:piece]
        center_piece = threshed_eye[0:h, piece: piece+piece]
        left_piece = threshed_eye[0:h, piece +piece:w]

        # calling pixel counter function
        eye_position = self.pixelCounter(right_piece, center_piece, left_piece)

        return eye_position 
    
    def pixelCounter(self, first_piece, second_piece, third_piece):
        # counting black pixel in each part 
        right_part = np.sum(first_piece==0)
        center_part = np.sum(second_piece==0)
        left_part = np.sum(third_piece==0)
        # creating list of these values
        eye_parts = [right_part, center_part, left_part]

        # getting the index of max values in the list 
        max_index = eye_parts.index(max(eye_parts))
        pos_eye ='' 
        if max_index==0:
            pos_eye="RIGHT"
            # color=[utils.BLACK, utils.GREEN]
        elif max_index==1:
            pos_eye = 'CENTER'
            # color = [utils.YELLOW, utils.PINK]
        elif max_index ==2:
            pos_eye = 'LEFT'
            # color = [utils.GRAY, utils.YELLOW]
        else:
            pos_eye="Closed"
            # color = [utils.GRAY, utils.YELLOW]
        # return pos_eye, color
        return pos_eye

            

