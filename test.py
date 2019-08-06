import numpy as np
import cv2
import  imutils
import pytesseract
#import pandas as pd
#import PIL
from PIL import Image
image = cv2.imread('car.jpeg')

image = imutils.resize(image, width=500)

cv2.imshow("Original Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("1 - Grayscale Conversion", gray)

gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# blurred = cv2.medianBlur(gray, 5)

# apply Canny Edge Detection
edged = cv2.Canny(blurred, 0, 50)
#orig_edged = edged.copy()
#cv2.imwrite('Edged/{}'.format(img), edged)

#edged = cv2.Canny(gray, 170, 200)
cv2.imshow("3 - Canny Edges", edged)

(new, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
NumberPlateCnt = None 

count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  
            NumberPlateCnt = approx 
            break

# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=mask)
cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
cv2.imshow("Final_image",new_image)
cv2.imwrite("ocrip.jpg",new_image)
#im = Image.open("ocrip.jpg")
#print(pytesseract.image_to_string(Image.open("ocrip.jpg"),lang="eng"))
#test=pytesseract.image_to_string(im,lang="eng")
#Configuration for tesseract
config = ('-l eng --oem 1 --psm 3')
#Run tesseract OCR on image
text = pytesseract.image_to_string(new_image, config=config)
print(text)
cv2.waitKey(0)
