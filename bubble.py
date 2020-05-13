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

def edge(image):
    canny =cv2.Canny(image,50,55)
    lap = cv2.Laplacian(image,cv2.CV_64F)#拉普拉斯边缘检测
    lap = np.uint8(np.absolute(lap))##对lap去绝对值
    cv2.namedWindow("edge",flags=cv2.WINDOW_AUTOSIZE)
    cv2.imshow("edge",canny)
    return canny

def close(edge):
    #kernel = np.ones((5,5),np.uint16)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    close = cv2.morphologyEx(edge,cv2.MORPH_CLOSE,kernel)
    #close = cv2.GaussianBlur(close,(7,7),0)
    cv2.namedWindow("close",flags=cv2.WINDOW_AUTOSIZE)
    cv2.imshow("close",close)
    return close
#circle =  cv2.HoughCircles ()

def circle_detection(close,grey):
    grey = cv2.cvtColor(grey,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(close,cv2.HOUGH_GRADIENT,1,15,
                            param1=100,param2=5,minRadius=1,maxRadius=10)
    # circles = np.uint16(np.around(circles))
    if circles is not None:
        for i in circles[0,:]:
        # draw the outer circle
            cv2.circle(grey,(i[0],i[1]),i[2],(0,0,255),1)
            # draw the center of the circle
            #cv2.circle(grey,(i[0],i[1]),1,(0,0,255),3)
        cv2.namedWindow("circle",flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow("circle",grey)
        cv2.waitKey(0)
        return grey

# def enhance_contrast(image):
#     hist = cv2.calcHist([image],[0],mask=None , histSize = 256, ranges = [0,256])





def sliding_window(grey,window_size = [160,160],hop_length = 40):
    crop_image = []
    l = window_size[0]
    h = window_size[1]
    image_with_circles = []
    print(len(grey[0]))
    for i in range((grey.shape[0]-window_size[0])//hop_length+1):
        for j in range((grey.shape[1]-window_size[1])//hop_length+1):
            image = grey[i*hop_length :i*hop_length + l , j*hop_length:j*hop_length+h].copy()
            crop_image.append(image)
            edges = edge(image)
            image = circle_detection(edges,image)


        # image_with_circles = list(filter(None , image_with_circles))
        #  cv2.namedWindow("1")
        #  cv2.imshow("1",)
        #  cv2.waitKey(0)
    return crop_image






if __name__ == "__main__":
    image = cv2.imread("./bubble/data/IMG_5672.jpeg")
    image = cv2.resize(image,(1280,960))
    image = image[360:720,360:720]
    #cv2.imwrite("./bubble/RGB.jpg",image)      
    grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)   
    grey = cv2.equalizeHist(grey)
    grey = cv2.GaussianBlur(grey,(11,11),0)
    #cv2.imwrite("./bubble/grey.jpg",grey)   
    #crop_image = sliding_window(grey)

    # cv2.namedWindow("image",flags=cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("image",image)
    # cv2.namedWindow("grey",flags=cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("grey",grey)
    sharped = sharp(grey)
    edge_0 = edge(sharped)
    #cv2.imwrite("./bubble/edge_0.jpg",edge_0) 
    edge_1 = edge(grey)
    #cv2.imwrite("./bubble/edge_1.jpg",edge_1) 
    inverse = 255-edge_0           
    close = close(edge_1)
    #open_opeation = open_opeation()         
    grey = circle_detection(grey,edge_1)
    #cv2.imwrite("./bubble/close.jpg",close) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()