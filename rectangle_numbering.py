import cv2

image= cv2.imread('img.png')
gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges= cv2.Canny(gray, 50,200)
img_gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh2 = cv2.threshold(img_gray2, 150, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

def get_contour_areas(contours):
    all_areas= []
    for cnt in contours:
        area= cv2.contourArea(cnt)
        all_areas.append(area)

    return all_areas

sorted_contours= sorted(contours, key=cv2.contourArea, reverse= False)
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(0,4):
    approx = cv2.approxPolyDP(sorted_contours[i], 0.009 * cv2.arcLength(contours[i], True), True)
    n = approx.ravel()
    x=n[-2]
    y=n[-1]+75
    cv2.putText(image, str(i+1), (x,y),font, 1, (0, 0, 0), 2, cv2.LINE_AA)



cv2.imshow("output",image)
cv2.imwrite("output1.png",image)
cv2.waitKey(0)
