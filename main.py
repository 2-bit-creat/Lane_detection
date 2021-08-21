import cv2
import numpy as np
import serial
import matplotlib.pyplot as plt
import time
from get_encoder import run
from set_encoder import writeBuffer, serWrite

class LaneDetection:

    low_yellow = np.array([15, 150, 120]) #hsv
    high_yellow = np.array([25, 230, 210])
    low_white = np.array([0, 150, 0]) #hsl
    high_white = np.array([255, 255, 255])

    def run(self, vid):
        cap = cv2.VideoCapture(vid)
        while cap.isOpened:
            _, frame = cap.read()
            frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            colored_image = self.color_detect(frame)
            cropped_image = self.cropping_image(colored_image)
            bird_eye_image = self.bird_eye(cropped_image)
            self.slidng_list_left = self.sliding_left(bird_eye_image) #return list of tuples(avg points of left side)
            self.sliding_list_right = self.sliding_right(bird_eye_image) #return list of tuples(avg points of right side)
            frame_steer = self.steering(frame) #print steering value

            bird_eye_image = cv2.cvtColor(bird_eye_image, cv2.COLOR_GRAY2BGR)
            for i in self.slidng_list_left:
                cv2.circle(bird_eye_image, i, 5, (255, 255, 0), -1)
            for i in self.sliding_list_right:
                cv2.circle(bird_eye_image, i, 5, (255, 255, 0), -1)

            cv2.imshow("bird_eye", bird_eye_image)
            cv2.imshow("frame", frame_steer)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def steering(self, frame):
        x_left = []
        x_right = []
        for i in range(0, len(self.slidng_list_left)): #extract x coordinate of left side
            x_left.append(self.slidng_list_left[i][0])
        left_arr = np.array(x_left)
        left_diff_arr = np.diff(x_left)
        left_diff_sum = np.sum(left_diff_arr)
        left_avg = int(left_diff_sum / 5) #avg slope values of left side

        for i in range(0, len(left_diff_arr)):
            if (left_diff_arr[i] > (left_avg +10)) or (left_diff_arr[i] < (left_avg - 10)): # remove inappropriate cases
                self.slidng_list_left = []
                left_diff_sum = 0

        for i in range(0, len(self.sliding_list_right)): #extract x coordinate of right side
            x_right.append(self.sliding_list_right[i][0])
        right_arr = np.array(x_right)
        right_diff_arr = np.diff(x_right)
        right_diff_sum = np.sum(right_diff_arr)
        right_avg = int(right_diff_sum / 5) #avg slope values of right side

        for i in range(0, len(right_diff_arr)):
            if (right_diff_arr[i] > (right_avg +10)) or (right_diff_arr[i] < (right_avg - 10)): # remove inappropriate cases
                self.sliding_list_right = []
                right_diff_sum = 0

        avg_val = int((left_diff_sum + right_diff_sum)/2) #avg of the avg slope valuses (dx/dy) of each side
        print(avg_val)
        if np.sum(avg_val) < -7: #turn right
            cv2.arrowedLine(frame, (300, 340), (340, 340), (255, 0, 0), 4)
        elif np.sum(avg_val) > 7: #turn left
            cv2.arrowedLine(frame, (340, 340), (300, 340), (255, 0, 0), 4)
        else: #go straight
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

    def sliding_left(self, img):
        left_list = []
        for j in range(259, img.shape[0] - 20, 40): #row
            j_list = []
            for i in range(19, int(img.shape[1]/2) - 20, 5): #col left side
                num_sum = np.sum(img[j - 19:j + 21, i - 19:i + 21]) #window size is 20*20
                if num_sum > 40000: #pick i given j where its num_sum is over 40000
                    j_list.append(i)
            try:
                len_list = []
                result = np.split(j_list, np.where(np.diff(j_list) > 5)[0] + 1) #cluster in case a gap b/w elements is over 5

                for k in range(0, len(result)):
                    len_list.append(len(result[k]))
                largest_integer = max(len_list)

                for l in range(0, len(result)):
                    if len(result[l]) == largest_integer:
                        avg = int(np.sum(result[l]) / len(result[l]))  # average
                        left_list.append((avg, j))  # avg points of left side
            except:
                continue
        return left_list

    def sliding_right(self, img):
        right_list = []
        for j in range(259, img.shape[0] - 20, 40): #row
            j_list = []
            for i in range(int(img.shape[1]/2), img.shape[1] - 20, 5): #col right side
                num_sum = np.sum(img[j - 19:j + 21, i - 19:i + 21]) #window size is 20*20
                if num_sum > 40000: #pick i given j where its num_sum is over 40000
                    j_list.append(i)
            try:
                len_list = []
                result = np.split(j_list, np.where(np.diff(j_list) > 5)[0] + 1) #cluster in case a gap b/w elements is over 5

                for k in range(0, len(result)):
                    len_list.append(len(result[k]))
                largest_integer = max(len_list)

                for l in range(0, len(result)):
                    if len(result[l]) == largest_integer:
                        avg = int(np.sum(result[l]) / len(result[l]))  # average
                        right_list.append((avg, j))  # avg points of left side
            except:
                continue
        return right_list

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
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        mask_yellow = cv2.inRange(hsv, self.low_yellow, self.high_yellow)
        mask_white = cv2.inRange(hls, self.low_white, self.high_white)
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        return mask

    def bird_eye(self, frame):
        frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        pts1 = np.float32([[180, 339], [410, 339], [90, 419], [490, 419]])
        pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, (640, 480))
        return result

if __name__ == '__main__':
    a = LaneDetection()
    a.run('road_vid2_Trim.mp4')

