import cv2
import numpy as np



kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
kernel_sharpen_2 = np.array([
        [1,1,1],
        [1,-9,1],
        [1,1,1]])
kernel_sharpen_3 = np.array([
        [-1,-1,-1,-1,-1],
        [-1,2,2,2,-1],
        [-1,2,8,2,-1],
        [-1,2,2,2,-1], 
        [-1,-1,-1,-1,-1]])/8.0


def sharp(grey):
    sharped = cv2.filter2D(grey,-1,kernel_sharpen_3)
    cv2.namedWindow("sharpen",flags=cv2.WINDOW_AUTOSIZE)
    cv2.imshow("sharpen",sharped)
    return sharped

def edge(grey,sharpen):
    canny =cv2.Canny(sharpen,50,55)
    lap = cv2.Laplacian(grey,cv2.CV_64F)#拉普拉斯边缘检测
    lap = np.uint8(np.absolute(lap))##对lap去绝对值
    cv2.namedWindow("edge",flags=cv2.WINDOW_AUTOSIZE)
    cv2.imshow("edge",canny)
    return canny

def close(edge):
    #kernel = np.ones((5,5),np.uint16)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    close = cv2.morphologyEx(edge,cv2.MORPH_CLOSE,kernel)
    cv2.namedWindow("close",flags=cv2.WINDOW_AUTOSIZE)
    cv2.imshow("close",close)
    return close
#circle =  cv2.HoughCircles ()

def circle_detection(close,grey):
    circles = cv2.HoughCircles(close, cv2.HOUGH_GRADIENT,1,100,
                            param1=100,param2=50,minRadius=10,maxRadius=50)
    # circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(grey,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(grey,(i[0],i[1]),2,(0,0,255),3)
        cv2.namedWindow("circle",flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow("circle",grey)
    return grey


if __name__ == "__main__":
    image = cv2.imread("./bubble/data/IMG_5672.jpeg")
    image = cv2.resize(image,(1280,960))
    grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # cv2.namedWindow("image",flags=cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("image",image)
    # cv2.namedWindow("grey",flags=cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("grey",grey)

    sharped = sharp(grey)
    edge = edge(grey,sharped)
    close = close(edge)
    grey = circle_detection(sharped,grey)
    cv2.waitKey(0)
    cv2.destroyAllWindows()