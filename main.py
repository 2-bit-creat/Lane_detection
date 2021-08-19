import cv2
import numpy as np
import serial
import matplotlib.pyplot as plt
import time
from get_encoder import run
from set_encoder import writeBuffer, serWrite

class LaneDetection:

    low_yellow = np.array([15, 150, 120])
    high_yellow = np.array([25, 230, 210])
    low_white = np.array([0, 0, 70])
    high_white = np.array([3, 60, 230])
    #low_white_hls = np.array([0, 153, 0])
    #high_white_hls = np.array([22, 179, 13])
    # low_white_rgb = np.array([160, 140, 140])
    # high_white_rgb = np.array([175, 170, 170])

    def run(self, vid):
        cap = cv2.VideoCapture(vid)
        while cap.isOpened:
            _, frame = cap.read()
            frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            colored_image = self.color_detect(frame)
            cropped_image = self.cropping_image(colored_image)
            bird_eye_image = self.bird_eye(cropped_image)
            slidng_list_left, sliding_list_right = self.sliding(bird_eye_image) #return two lists comprised of tuples
            frame_steer = self.steering(frame, slidng_list_left, sliding_list_right) #print steering direction

            bird_eye_image = cv2.cvtColor(bird_eye_image, cv2.COLOR_GRAY2BGR)
            for i in slidng_list_left:
                cv2.circle(bird_eye_image, i, 5, (255, 255, 0), -1)
            for i in sliding_list_right:
                cv2.circle(bird_eye_image, i, 5, (255, 255, 0), -1)

            cv2.imshow("bird_eye", cropped_image)
            cv2.imshow("frame", frame_steer)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def steering(self, frame, sliding_left, sliding_right):
        x_left = []
        x_right = []

        for i in range(0, len(sliding_left)): #extract x coordinate
            x_left.append(sliding_left[i][0])
        left_arr = np.array(x_left)

        for i in range(0, len(sliding_right)): #extract x coordinate
            x_right.append(sliding_right[i][0])
        right_arr = np.array(x_right)

        avg_val = int((np.sum(np.diff(x_left))+np.sum(np.diff(x_right)))/2)
        print(avg_val)
        if np.sum(avg_val) < -7:
            cv2.arrowedLine(frame, (300, 340), (340, 340), (255, 0, 0), 4)
        elif np.sum(avg_val) > 7:
            cv2.arrowedLine(frame, (340, 340), (300, 340), (255, 0, 0), 4)
        else:
            cv2.arrowedLine(frame, (320, 340), (320, 300), (255, 0, 0), 4)

        return frame

    def simulation(self, vid):
        cap = cv2.VideoCapture(vid)
        while True:
            _, img = cap.read()
            img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            for j in range(259, img.shape[0] - 20, 40): #row
                for i in range(19, img.shape[1] - 20, 20): #col
                    cv2.rectangle(img, (i-19,j-19),(i+20,j+20), (255,255,255),3)
                    cv2.imshow("img", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    def sliding(slef, img):
        left_list = []
        right_list = []
        for j in range(259, img.shape[0] - 20, 40): #row
            j_list = []
            for i in range(19, img.shape[1] - 20, 5): #col
                num_sum = np.sum(img[j - 9:j + 11, i - 9:i + 11]) #window size is (10,10)
                if num_sum > 4000: #pick i given j where its num_sum is over 4000
                    j_list.append(i)
            try:
                result = np.split(j_list, np.where(np.diff(j_list) > 5)[0] + 1) #clustering
                for k in range(0, len(result)):
                    if len(result[k]) < 5:
                        continue
                    avg = int(np.sum(result[k]) / len(result[k])) #average
                    if avg < 320:
                        left_list.append((avg, j)) #avg points of left side
                    else:
                        right_list.append((avg, j)) #avg points of right side
            except:
                continue
        return left_list, right_list

    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        match_mask_color = (255,255,255)
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def cropping_image(self, image):
        height = image.shape[0]
        width = image.shape[1]
        region_of_interest_vertices = [
            (0, height),
            (width/2, height/2),
            (width, height)
        ]
        cropped_image = self.region_of_interest(image,
                        np.array([region_of_interest_vertices], np.int32),)
        return cropped_image

    def color_detect(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(hsv, self.low_yellow, self.high_yellow)
        mask_white = cv2.inRange(hsv, self.low_white, self.high_white)
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        return mask

    def bird_eye(self, frame):
        frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        # cv2.circle(frame, (170, 339), 5, (0, 0, 255), -1)
        # cv2.circle(frame, (420, 339), 5, (0, 0, 255), -1)
        # cv2.circle(frame, (90, 419), 5, (0, 0, 255), -1)
        # cv2.circle(frame, (490, 419), 5, (0, 0, 255), -1)
        #cv2.imshow("frame", frame)
        pts1 = np.float32([[180, 339], [410, 339], [90, 419], [490, 419]])
        pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, (640, 480))
        return result

if __name__ == '__main__':
    #seri = serial.Serial("COM3", 115200)
    a = LaneDetection()
    a.run('road_vid2_Trim.mp4')
    # cap = cv2.VideoCapture('road_vid2_Trim.mp4')
    # while True:
    #     _, frame = cap.read()
    #     cv2.imshow("bird", a.bird_eye(frame))
    #     cv2.waitKey(1)

