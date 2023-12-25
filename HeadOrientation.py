import cv2
import numpy as np
import pygame
from pygame import mixer
import time

# from Utils import rotationMatrixToEulerAngles, draw_pose_info

class HeadOrientation:
    def __init__(self,looking_left, looking_right, looking_up,looking_down,forward,mixer,camera_matrix = None, dist_coeffs = None, show_axis : bool = False):
        """
        This class contains methods to calculated the angles(roll, pitch, yaw) of the head. It optionally also uses the camera parameters
        
        Camera_matrix : matrix of camera used to capture the image/frame

        dist_coeffs : distortion coefficients of the camera used to capture the image/frame

        
        """
        self.looking_up = looking_up
        self.looking_down = looking_down
        self.looking_right = looking_right
        self.looking_left = looking_left
        self.forward = forward
        self.show_axis = show_axis
        self.camera_matrix = camera_matrix
        self.sound_counter = 0
        self.dist_coeffs = dist_coeffs
        self.current_time = 0
        self.previous_time = 0    
    
    def get_pose(self, frame, landmarks, width, height):
        """
        Estimate head pose using the head pose estimator object instantiated attribute
        Parameters
        ----------
        frame: numpy array
            Image/frame captured by the camera
        landmarks: dlib.rectangle
            Dlib detected 68 landmarks of the head
        Returns
        --------
        - if successful: image_frame, roll, pitch, yaw (tuple)
        - if unsuccessful: None,None,None,None (tuple)
        """
        self.model_index = [33,263,1,61,291,199]
        self.keypoints = landmarks
        self.frame = frame  # opencv image array
        # self.image_points = np.array([])

        self.axis = np.float32([[200, 0, 0],
                                [0, 200, 0],
                                [0, 0, 200]])
        # array that specify the length of the 3 projected axis from the nose

        if self.camera_matrix is None:
            # if no camera matrix is given, estimate camera parameters using picture size
            self.size = frame.shape
            self.focal_length = self.size[1]
            self.center = (self.size[1] / 2, self.size[0] / 2)
            self.camera_matrix = np.array(
                [[self.focal_length, 0, self.center[0]],
                 [0, self.focal_length, self.center[1]],
                 [0, 0, 1]], dtype="double"
            )

        if self.dist_coeffs is None:  # if no distorsion coefficients are given, assume no lens distortion
            self.dist_coeffs = np.zeros((4, 1))

        # 3D Head model world space points (generic human head)
        # self.model_points = np.array([
        #     (0.0, 0.0, 0.0),  # Nose tip
        #     (0.0, -330.0, -65.0),  # Chin
        #     (-225.0, 170.0, -135.0),  # Left eye left corner
        #     (225.0, 170.0, -135.0),  # Right eye right corner
        #     (-150.0, -150.0, -125.0),  # Left Mouth corner
        #     (150.0, -150.0, -125.0)  # Right mouth corner

        # ])

        face_2d  = []
        face_3d = []
        
        if landmarks:
            for face_landmarks in landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * width, lm.y * height)
                            nose_3d = (lm.x * width, lm.y * height, lm.z * 8000)

                        x, y = int(lm.x * width), int(lm.y * height)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * width

                cam_matrix = np.array([ [focal_length, 0, height / 2],
                                        [0, focal_length, height / 2],
                                        [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # print(y)

                # See where the user's head tilting
                if y < -10:
                    text = "Looking-Left"
                elif y > 10:
                    text = "Looking-Right"
                elif x < -10:
                    text = "Looking-Down"
                elif x > 10:
                    text = "Looking-Up"
                else:
                    text = "Forward"

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                
                # cv2.line(frame, p1, p2, (255, 0, 0), 5)

                # Add the text on the image
                # cv2.putText(frame, text, (20, 1500), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 5)
                cv2.putText(frame, text, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, f"x : {round(x,2)}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, f"y : {round(y, 2)}", (10,190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, f"z : {round(z,2)}", (10,230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                self.current_time = time.time()
                # print("The previous time is ", self.previous_time)
                # print("The current time is ", self.current_time)
                if (self.current_time - self.previous_time > 5) or self.previous_time == 0:
                    if text == "Looking-Up":
                        if mixer.get_busy() == 0:
                            self.forward.play()
                    elif text == "Looking-Down":
                        if mixer.get_busy() == 0:
                            self.forward.play()
                    elif text == "Looking-Right":
                        if mixer.get_busy() == 0:
                            self.forward.play()
                    elif text == "Looking-Left":
                        if mixer.get_busy() == 0:
                            self.forward.play()

                    self.previous_time = self.current_time

                return self.frame, x, y, z
        else:
            return None, None, None, None