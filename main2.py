import cv2 as cv
import mediapipe as mp
import time

class HandArmDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLM in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLM, self.mpHands.HAND_CONNECTIONS)

        return img

    def findHandArmPosition(self, img, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList

def main():
    pTime = 0
    cap = cv.VideoCapture(0)
    detector = HandArmDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        try:
            hand_arm_landmarks = detector.findHandArmPosition(img)
            if hand_arm_landmarks:
                print("Hand and Arm Landmarks:")
                for lm in hand_arm_landmarks:
                    print(f"Landmark {lm[0]}: ({lm[1]}, {lm[2]})")
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
