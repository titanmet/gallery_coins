# posts/models.py
from django.db import models
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from skimage import color


class Post(models.Model):
    title = models.TextField()
    cover = models.ImageField()
    info = models.TextField()

    def __str__(self):
        return self.title

    def detect_coins(self):
        coins = cv2.imread(str(self.cover), 1)
        gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            1,
            50,
            param1=100,
            param2=50,
            minRadius=10,
            maxRadius=380,
        )

        coins_copy = coins.copy()

        for detected_circle in circles[0]:
            x_coor, y_coor, detected_radius = detected_circle
            coins_detected = cv2.circle(
                coins_copy,
                (int(x_coor), int(y_coor)),
                int(detected_radius),
                (0, 255, 0),
                4,
            )

        cv2.imwrite("test_Hough.jpg", coins_detected)

        return circles

    @property
    def calculate_amount(self):
        rub = {
            "1 RUB": {
                "value": 1,
                "radius": 20.5,
                "ratio": 1,
                "count": 0,
            },
            "2 RUB": {
                "value": 2,
                "radius": 23,
                "ratio": 1.15,
                "count": 0,
            },
            "5 RUB": {
                "value": 5,
                "radius": 25,
                "ratio": 1.25,
                "count": 0,
            },
            "10 RUB": {
                "value": 10,
                "radius": 22,
                "ratio": 1.07,
                "count": 0,
            },
        }

        circles = self.detect_coins()
        radius = []
        coordinates = []

        for detected_circle in circles[0]:
            x_coor, y_coor, detected_radius = detected_circle
            radius.append(detected_radius)
            coordinates.append([x_coor, y_coor])

        smallest = min(radius)
        tolerance = 0.0375
        total_amount = 0

        coins_circled = cv2.imread('test_Hough.jpg', 1)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for coin in circles[0]:
            ratio_to_check = coin[2] / smallest
            coor_x = coin[0]
            coor_y = coin[1]
            for rubli in rub:
                value = rub[rubli]['value']
                if abs(ratio_to_check - rub[rubli]['ratio']) <= tolerance:
                    rub[rubli]['count'] += 1
                    total_amount += rub[rubli]['value']
                    cv2.putText(coins_circled, str(value), (int(coor_x), int(coor_y)), font, 1,
                                (0, 0, 0), 4)

        return total_amount

    @property
    def calculate_coins(self):
        rub = {
            "1 RUB": {
                "value": 1,
                "radius": 20.5,
                "ratio": 1,
                "count": 0,
            },
            "2 RUB": {
                "value": 2,
                "radius": 23,
                "ratio": 1.15,
                "count": 0,
            },
            "5 RUB": {
                "value": 5,
                "radius": 25,
                "ratio": 1.25,
                "count": 0,
            },
            "10 RUB": {
                "value": 10,
                "radius": 22,
                "ratio": 1.07,
                "count": 0,
            },
        }

        circles = self.detect_coins()
        radius = []
        coordinates = []

        for detected_circle in circles[0]:
            x_coor, y_coor, detected_radius = detected_circle
            radius.append(detected_radius)
            coordinates.append([x_coor, y_coor])

        smallest = min(radius)
        tolerance = 0.0375
        total_amount = 0
        total_count = 0

        coins_circled = cv2.imread('test_Hough.jpg', 1)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for coin in circles[0]:
            ratio_to_check = coin[2] / smallest
            coor_x = coin[0]
            coor_y = coin[1]
            for rubli in rub:
                value = rub[rubli]['value']
                if abs(ratio_to_check - rub[rubli]['ratio']) <= tolerance:
                    rub[rubli]['count'] += 1
                    total_amount += rub[rubli]['value']
                    cv2.putText(coins_circled, str(value), (int(coor_x), int(coor_y)), font, 1,
                                (0, 0, 0), 4)

        for rubli in rub:
            total_count += rub[rubli]['count']

        return total_count

    def size_image(self):
        coins = cv2.imread(str(self.cover), 1)
        gray_im = color.rgb2gray(coins)
        thresh = threshold_otsu(gray_im, nbins=5)
        thresholded = gray_im > thresh
        no_small = morphology.remove_small_objects(thresholded, min_size=150)
        img = morphology.binary_closing(no_small, disk(3))
        distance_im = ndi.distance_transform_edt(img)
        return distance_im.shape

    def avr_rgb(self):
        from PIL import Image, ImageStat
        img = Image.open(self.cover)
        mean = ImageStat.Stat(img).mean
        return '({:.0f}, {:.0f}, {:.0f})'.format(*mean)


