import cv2 as cv
import mediapipe as mp
import time
import math

class HandArmDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=2,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPoseLandmarks(self, img):
        landmark_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w) 
                landmark_list.append([id, cx, cy, cz])
        return landmark_list

    def calculate_angle(self, p1, p2, p3):
        # Calculate angle between three points using arctan2
        delta_x1 = p1[1] - p2[1]
        delta_y1 = p1[2] - p2[2]
        delta_x3 = p3[1] - p2[1]
        delta_y3 = p3[2] - p2[2]

        angle_rad1 = math.atan2(delta_y1, delta_x1)
        angle_rad3 = math.atan2(delta_y3, delta_x3)

        angle_rad = angle_rad3 - angle_rad1
        angle_deg = math.degrees(angle_rad)
        return angle_deg

def main():
    pTime = 0
    cap = cv.VideoCapture(0)
    detector = HandArmDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        try:
            pose_landmarks = detector.findPoseLandmarks(img)
            if pose_landmarks and len(pose_landmarks) >= 3:  # Ensure necessary landmarks exist
                shoulder_angle_right = detector.calculate_angle(pose_landmarks[24], pose_landmarks[12], pose_landmarks[14])
                elbow_angle_right = detector.calculate_angle(pose_landmarks[16], pose_landmarks[14], pose_landmarks[12])
                wrist_angle_right = detector.calculate_angle(pose_landmarks[20], pose_landmarks[16], pose_landmarks[14])

                print(f"Shoulder Angle Right: {shoulder_angle_right:.2f} degrees")
                print(f"Elbow Angle Right: {elbow_angle_right:.2f} degrees")
                print(f"Wrist Angle Right: {wrist_angle_right:.2f} degrees")
        except Exception as ex:
            print(f'An Exception Occurred: {ex}')
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow('image', img)
        k = cv.waitKey(1)
        if k == 27:
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
