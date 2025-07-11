import cv2
import numpy as np
import utils


webCamFeed = True
# pathImage = "document_image.jpg"
pathImage = r"C:\Users\Welcome\OpenCV Projects\Project-1_DocumentScanner\document_image.jpg"
capture = cv2.VideoCapture(0)
capture.set(10,160)
heightImg = 640
widthImg = 480


utils.initializeTrackbars()
count=0

while True:
    blank = np.zeros((heightImg,widthImg,3),np.uint8)
    img = cv2.imread(pathImage)
    #Get the image
    # if webCamFeed:
    #     success , img = capture.read()
    # else:
        # img = cv2.imread(pathImage)

    #Preprocessing of the image
    img = cv2.resize(img , (widthImg,heightImg))
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),1)
    thresh = utils.valTrackbars()
    imgThreshold = cv2.Canny(blur , thresh[0] , thresh[1])
    kernel = np.ones((5,5))
    dial = cv2.dilate(imgThreshold , kernel , iterations=2)
    imgThreshold = cv2.erode(dial,kernel,iterations=1)

    #Find all the contours
    imgContours = img.copy()
    imgBigContour = img.copy()
    contours , hierarchy = cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours,contours,-1,(0,255,0),10)

    #Find the biggest contour
    biggest , maxArea = utils.biggestContour(contours)
    if biggest.size != 0:
        biggest = utils.reorder(biggest)
        cv2.drawContours(imgBigContour,biggest,-1,(0,255,0),20)
        imgBigContour = utils.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        imgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

        #Remove 20 Pixels from each side
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20 , 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))
    
        #Apply Adaptive Threshold (Post-Processing)
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThresh = cv2.adaptiveThreshold(imgWarpGray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,2)
        imgAdaptiveThresh = cv2.bitwise_not(imgAdaptiveThresh)
        imgAdaptiveThresh = cv2.medianBlur(imgAdaptiveThresh,3)
    
        #Image Array for Display
        imageArray = ([img,gray,imgThreshold,imgContours],[imgBigContour,imgWarpColored,imgWarpGray,imgAdaptiveThresh])
    
    else:
        imageArray = ([img,gray,imgThreshold,imgContours],[blank,blank,blank,blank])

    labels = [["Original","Gray","Threshold","Contours"],["Biggest Contour","Warp Perspective","Warp Gray","Adaptive Threshold"]]

    stackedImage = utils.stackImages(imageArray,0.75,labels)
    cv2.imshow("Result",stackedImage)

    #Save image when 's' is pressed 
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myimage" + str(count)+".jpg",imgWarpColored)
        cv2.rectangle(stackedImage,((int(stackedImage.shape[1]/2) - 230) , (int(stackedImage.shape[0]/2) - 230) , (1100,350) , (0,255,0) , cv2.FILLED))
        cv2.putText(stackedImage,"Scan Saved" , (int(stackedImage.shape[1]/2) - 200) , (int(stackedImage.shape[0]/2) - 200) , cv2.FONT_HERSHEY_COMPLEX , 3 , (0,0,255) , 5 , cv2.LINE_AA)
        cv2.imshow('Result',stackedImage)
        cv2.waitKey(300)
        count+=1