import cv2
import matplotlib
#This is function is only for binary masks , like an example shown in the README.md with pixel values {0,255}

def BoundingBoxGenerator(ImagePath): #Image to be fed in as numpy array
	imago = cv2.imread(ImagePath , cv2.IMREAD_COLOR)
	matplotlib.pyplot.imshow(imago)
	ret,thresh = cv2.threshold(imago,0.001,255,cv2.THRESH_BINARY) #For Sanity
	im2 ,contours, hierarchy = cv2.findContours(cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY).copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]
	x,y,w,h = cv2.boundingRect(cnt)
	#x,y are bottom left coordinates of box with origin are upper left of the image and w ,h are width and height respectively
	return x,y,w,h
