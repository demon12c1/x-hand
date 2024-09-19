import cv2
import mediapipe as mp
import time
import requests
import os
from dotenv import load_dotenv
import threading
load_dotenv()

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.1, trackCon=0.1):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        # Load API endpoint from environment variables
        self.apiEndpoint = os.getenv("API_ENDPOINT")
        self.last_api_call = 0

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        if lmList and self.canCallAPI():
            threading.Thread(target=self.callExternalAPI, args=()).start()

        return lmList

    def canCallAPI(self):
        """Checks if 5 seconds have passed since the last API call."""
        current_time = time.time()
        if current_time - self.last_api_call >= 5:  # 5 seconds
            self.last_api_call = current_time
            return True
        return False

    def callExternalAPI(self):
        # Example API call
        if not self.apiEndpoint:
            print("API endpoint is not set in the environment file!")
            return

            # Example API call
        data = {
            "machine_code": 'MCL001_MC001',
            "operator_id": 12,
            "operator_name": 'Nguyen Van A',
            "working_time": time.time()
        }

        try:
            response = requests.post(self.apiEndpoint, json=data)
            print(f"API Response: {response.status_code}, {response.json()}")
        except Exception as e:
            print(f"Failed to call API: {e}")


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()

    while True:
        success, img = cap.read()

        # glove = detector.findGlove(img, lower_green, upper_green)
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
