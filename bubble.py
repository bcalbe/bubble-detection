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
    canny =cv2.Canny(image,50,100)
    lap = cv2.Laplacian(image,cv2.CV_64F)#拉普拉斯边缘检测
    lap = np.uint8(np.absolute(lap))##对lap去绝对值
    cv2.namedWindow("edge",flags=cv2.WINDOW_AUTOSIZE)
    cv2.imshow("edge",canny)
    return canny

def close(edge):
    #kernel = np.ones((5,5),np.uint16)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    close = cv2.morphologyEx(edge,cv2.MORPH_CLOSE,kernel_close)
    #close = cv2.morphologyEx(close,cv2.MORPH_OPEN,kernel_open)
    #close = cv2.GaussianBlur(close,(7,7),0)
    cv2.namedWindow("close",flags=cv2.WINDOW_AUTOSIZE)
    cv2.imshow("close",close)
    return close
#circle =  cv2.HoughCircles ()

def circle_detection(close,grey):
    circles = cv2.HoughCircles(close,cv2.HOUGH_GRADIENT,1,17,
                            param1=100,param2=5,minRadius=1,maxRadius=10)
    #cir = circles[0,20].astype(int)
    get_eclipse(grey,circles)

    grey = cv2.cvtColor(grey,cv2.COLOR_GRAY2BGR)
    
    
    if circles is not None:
        for i in circles[0,:]:
        # draw the outer circle
            cv2.circle(grey,(i[0],i[1]),i[2],(0,0,255),1)
            # draw the center of the circle
            #cv2.circle(grey,(i[0],i[1]),1,(0,0,255),3)
        #cv2.circle(grey,(cir[0],cir[1]),cir[2],(255,0,),1)
        cv2.namedWindow("circle",flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow("circle",grey)
        cv2.waitKey(0)
        return grey

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

def contour(edge_map):
    cv2.namedWindow("contour",flags=cv2.WINDOW_AUTOSIZE)
    
    image,cons,hierarchy=cv2.findContours(edge_map,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    if len(cons)>0:
        cnt = cons[0]
        area = cv2.contourArea(cnt)
        length = cv2.arcLength(cnt,False)
        isclose = cv2.isContourConvex(cnt)
        print("area is {} length is {} and close is {}".format(area,length,isclose))
        cv2.drawContours(image,cons,-1,(0,255,0),1) 
        cv2.imshow("contour",image)
    return cons

def get_eclipse(img,circles):
    image = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.namedWindow("eclipse",flags=cv2.WINDOW_NORMAL)
    r = 10 
    for cir in circles[0,:]: 
        cir = cir.astype(int)
        crop_image = img[cir[1]-2*r : cir[1]+ 2*r,cir[0]-2*r : cir[0]+ 2*r]
        con = contour(crop_image)
        
        if len(con)>0:
            target_cons =np.array( [x for x in con if len(x)>10] )
            #target_cons = get_max_contour(con)
            for target_con in target_cons:
                cv2.circle(image,(cir[0],cir[1]),3,(0,0,255),2)
                #if isarc(target_con) is True:
                if True:
                    #target_con = target_con.squeeze(0)
                    #y = target_con[:,:,:,0]+(cir[0]-2*cir[2])
                    x = target_con[:,:,0]+(cir[0]-2*r)
                    y = target_con[:,:,1]+(cir[1]-2*r)
                    target_con = np.append(x,y,axis=-1)
                    target_con = target_con[:,np.newaxis,:]
                    S1 = cv2.contourArea(target_con)
                    # if (rec[0][0]-rec[1][0])/(rec[0][1]-rec[1][1]) >2:
                    # cv2.rectangle(image,rec[0],rec[1],(255,0,0),3)
                    cv2.drawContours(image,target_con,-1,(0,255,0),1)
                    cv2.imshow("eclipse", image)
                    #cv2.waitKey(0)
    #cv2.imwrite("./bubble/5667_2_result_contour_blue.jpg",image) 
     

def get_max_contour(con):
    length = 0
    for x in con:
        if len(x)>length:
            length = len(x)
            target_con = x
    return target_con

def isarc(con):
    # length = len(con)
    # point1 = con[0,0]
    # point2 = con[int(length/2),0]
    # point3 = con[-1,0]
    # angle1 =np.arctan2((point2[0]-point1[0]),(point2[1]-point1[1]))
    # angle2 =np.arctan2((point3[0]-point2[0]),(point3[1]-point2[1]))
    # minus = np.degrees(angle1-angle2)
    # if minus>90:
        # return True
    # else:
    #     return False
    lines = np.array(cv2.HoughLinesPointSet(con,3 ,8,0,360,1,0,np.pi/2,np.pi/180))
    if lines.all():
        return True
    else:
        return False
    
def illumination(image):
    _, mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("original",flags = cv2.WINDOW_NORMAL)
    cv2.imshow("original",mask)
    cv2.waitKey(0)
    cv2.imshow("original",image)
    cv2.namedWindow("remove highlight",flags = cv2.WINDOW_NORMAL)
    image = cv2.inpaint(image,mask,20, cv2.INPAINT_TELEA)
    cv2.imwrite("./bubble/highlight_removal.jpg",image)
    cv2.imshow("remove highlight",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


if __name__ == "__main__":
    image = cv2.imread("./bubble/data/IMG_5672.jpeg")
    image = cv2.resize(image,(1280,960))
    #image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    B,G,R = cv2.split(image)
    B = image.copy()
    B[:,:,0] = 0
    B[:,:,1] = 0
    cv2.imwrite("./bubble/R.jpg",B)
    #image = illumination(image)
    cv2.namedWindow("Blue",flags = cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Blue",B)
    cv2.waitKey(0)
    cv2.destroyWindow("Blue")
    # R  = cv2.equalizeHist(R)
    # G  = cv2.equalizeHist(G)
    # B  = cv2.equalizeHist(B)
    # image = cv2.merge((B,G,R))
    #image = image[360:720,360:720]
    #cv2.imwrite("./bubble/RGB.jpg",image)      
    grey = B#cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)   
    grey = cv2.equalizeHist(grey)
    grey = cv2.GaussianBlur(grey,(15,15),0)
    grey = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 15)
    #grey = cv2.threshold(grey,150,255,cv2.THRESH_BINARY)
    #cv2.imwrite("./bubble/5667_2_blue_adaptive.jpg",grey)   
    #crop_image = sliding_window(grey)

    # cv2.namedWindow("image",flags=cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("image",image)
    # cv2.namedWindow("grey",flags=cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("grey",grey)
    sharped = sharp(grey)
    edge_0 = edge(sharped)
    #cv2.imwrite("./bubble/edge_0.jpg",edge_0) 
    edge_1 = edge(grey)
    #contour(edge_1)
    #cv2.imwrite("./bubble/edge_blue.jpg",edge_1) 
    inverse = 255-edge_0           
    close = close(grey)
    #open_opeation = open_opeation()         
    grey = circle_detection(grey,edge_1)
    #cv2.imwrite("./bubble/results_threshold_blue.jpg",grey) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()