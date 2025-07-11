import cv2
import numpy as np

def nothing(x):
    pass

def initializeTrackbars(initialTrackbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars",360,240)
    cv2.createTrackbar("Threshold1","Trackbars",200,255,nothing)
    cv2.createTrackbar("Threshold2","Trackbars",200,255,nothing)

def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1","Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2","Trackbars")
    src = Threshold1 , Threshold2
    return src

def biggestContour(contours):
    biggest = np.array([])
    max_area=0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.01*peri,True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest , max_area

def drawRectangle(img, points, thickness):
    cv2.line(img, tuple(points[0][0]), tuple(points[1][0]), (0, 255, 0), thickness)
    cv2.line(img, tuple(points[1][0]), tuple(points[3][0]), (0, 255, 0), thickness)
    cv2.line(img, tuple(points[3][0]), tuple(points[2][0]), (0, 255, 0), thickness)
    cv2.line(img, tuple(points[2][0]), tuple(points[0][0]), (0, 255, 0), thickness)
    return img

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),dtype=np.int32)
    add = myPoints.sum(1)
    
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints , axis=1)
    myPointsNew[2] = myPoints[np.argmax(diff)]
    myPointsNew[1] = myPoints[np.argmin(diff)]

    return myPointsNew

def stackImages(imgArray, scale, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    stackedImages = None

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        hor = [np.hstack(imgArray[x]) for x in range(rows)]
        stackedImages = np.vstack(hor)

        if labels:
            eachImgWidth = int(stackedImages.shape[1] / cols)
            eachImgHeight = int(stackedImages.shape[0] / rows)

            for d in range(rows):
                for c in range(cols):
                    cv2.rectangle(stackedImages,
                                  (c * eachImgWidth, eachImgHeight * d),
                                  (c * eachImgWidth + len(labels[d][c]) * 13 + 27, 30 + eachImgHeight * d),
                                  (255, 255, 255), cv2.FILLED)
                    cv2.putText(stackedImages, labels[d][c],
                                (c * eachImgWidth + 10, eachImgHeight * d + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return stackedImages
