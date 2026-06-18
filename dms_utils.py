import math
import numpy as np

# MediaPipe 468-Landmark Indices (Correct mapping for EAR/MAR)
# Right eye: outer corner, inner corner, upper1, upper2, lower2, lower1
RIGHT_EYE_LM = [33, 133, 160, 158, 153, 144]
# Left eye: outer corner, inner corner, upper1, upper2, lower2, lower1
LEFT_EYE_LM = [362, 263, 385, 387, 373, 380]

# Mouth landmarks
UPPER_LIP_LM = 13
LOWER_LIP_LM = 14
LEFT_MOUTH_LM = 78
RIGHT_MOUTH_LM = 308

def distance(pt1, pt2):
    return math.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1])

def calculate_EAR(landmarks, eye_indices):
    """
    Eye Aspect Ratio (EAR) for drowsiness detection.
    Index mapping: [outer, inner, upper1, upper2, lower2, lower1]
    EAR = avg(vertical) / horizontal
    When eyes are open: EAR ~ 0.25-0.35
    When eyes are closed: EAR ~ 0.05-0.15
    """
    p1, p4, p2, p3, p5, p6 = [landmarks[i] for i in eye_indices]
    hor = distance(p1, p4)                          # outer <-> inner
    ver = (distance(p2, p6) + distance(p3, p5)) / 2.0  # avg of two verticals
    return 0 if hor == 0 else ver / hor

def calculate_MAR(landmarks):
    """
    Mouth Aspect Ratio (MAR) for yawning detection.
    MAR = vertical / horizontal
    Normal: MAR ~ 0.1-0.3
    Yawning: MAR ~ 0.6+
    """
    mouth_up = landmarks[UPPER_LIP_LM]
    mouth_down = landmarks[LOWER_LIP_LM]
    mouth_left = landmarks[LEFT_MOUTH_LM]
    mouth_right = landmarks[RIGHT_MOUTH_LM]
    hor = distance(mouth_left, mouth_right)
    return 0 if hor == 0 else distance(mouth_up, mouth_down) / hor
