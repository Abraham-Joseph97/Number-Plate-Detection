import cv2
import numpy as np
import imutils
img=cv2.imread('LU14-023_1024x768.jpg',1)
img_1=cv2.imread("LU14-023_1024x768.jpg",0)
#cv2.imshow("img1",img_1)
image = imutils.resize(img_1, width=500)
image=cv2.getRectSubPix(img_1,(250,150),(150,98))
cv2.imshow("img2",image)
edges = cv2.Canny(img_1, 170, 200)
cv2.imshow("img3",edges)
(new, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None #we currently have no Number plate contour

# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            break


# Drawing the selected contour on the original image
cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)

cv2.waitKey(0) #Wait for user input before closing the images displayed`

cv2.destroyAllWindows()
