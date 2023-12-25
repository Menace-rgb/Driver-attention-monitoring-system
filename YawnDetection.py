import numpy as np
import cv2

class YawnDetection:
    def __init__(self):
        self.LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        self.UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
        self.bottom_lips_imp_coordinates = {}
        self.upper_lips_imp_coordinates = {}
        self.upper_coordinates = []
        self.bottom_coordinates = []

    def get_imp_coordinates(self,multi_face_landmarks, frame, width, height):
        if multi_face_landmarks:
            for faceLms in multi_face_landmarks:
                for id, lm in enumerate(faceLms.landmark):
                    if id in self.LOWER_LIPS:
                        req_x, req_y  = int(lm.x * width), int(lm.y * height)
                        cv2.circle(frame, (req_x, req_y), 1, (255, 0, 255), cv2.FILLED)
            

                        self.bottom_lips_imp_coordinates[id] = [req_x, req_y]
                        self.upper_coordinates.append((req_x, req_y))

                    if id in self.UPPER_LIPS:
                        req_x, req_y = int(lm.x * width), int(lm.y * height)
                        cv2.circle(frame,(req_x, req_y),1, (255, 0, 255), cv2.FILLED)

                        self.upper_lips_imp_coordinates[id] = [req_x, req_y]
                        self.bottom_coordinates.append((req_x, req_y))

    def get_distance(self):
        try:
            self.upper_coordinates = []
            self.bottom_coordinates = []

            for value in self.bottom_lips_imp_coordinates.values():
                self.bottom_coordinates.append(value)

            for value in self.upper_lips_imp_coordinates.values():
                self.upper_coordinates.append(value)
            bottom = np.mean(self.bottom_coordinates, axis = 0)
            # print(bottom)
            bottom_y = int(bottom[1])
            up = np.mean(self.upper_coordinates, axis = 0)
            up_y = int(up[1])
            return abs(up_y - bottom_y)
        except:
            None

    

