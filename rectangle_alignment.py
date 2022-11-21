import cv2
import numpy as np

image = cv2.imread('img.png')
imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
contours_to_rotate=[1,3,4,6,7,9,10,12]

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def rotate_contour(cnt, angle):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated

im_copy = image.copy()
im_copy.fill(255)
for i in contours_to_rotate:
    angle = cv2.minAreaRect(contours[i])[-1]
    approx = cv2.approxPolyDP(contours[i], 0.009 * cv2.arcLength(contours[i], True), True)
    n = approx.ravel()
    if n[0] > n[4] and n[1] < n[5]:
        angle = (270 - angle)
    else:
        angle = 360 - angle

    cnt_rotated = rotate_contour(contours[i], angle)
    cv2.drawContours(im_copy, [cnt_rotated], 0, (0, 0, 0), 4)


cv2.imshow("output",im_copy)
cv2.imwrite("output2.png",im_copy)
cv2.waitKey(0)
