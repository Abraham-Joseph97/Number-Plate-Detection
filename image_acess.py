import cv2
camera=cv2.VideoCapture(0)
cv2.namedWindow("cam test")
while 1:
        ret, im = camera.read()
        cv2.imshow("cam-test",im)
        cv2.imwrite("test.jpg",im)
del(camera)