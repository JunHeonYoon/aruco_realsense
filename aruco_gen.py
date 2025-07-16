import cv2
import numpy as np

if __name__ == "__main__":
    board_type=cv2.aruco.DICT_6X6_250
    MARKER_SIZE = 1200    
        
    for idx in range(8):
        arucoDict = cv2.aruco.getPredefinedDictionary(board_type)
        aruco_matker_img = cv2.aruco.generateImageMarker(arucoDict , idx , MARKER_SIZE)
        
        cv2.imshow(f"aruco_marker_img_{idx}",aruco_matker_img)
        cv2.imwrite(f"aruco/aruco_{idx}.png", aruco_matker_img)
        cv2.waitKey(0)