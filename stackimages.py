import numpy as np
import cv2

def stackImages(imgArray,scale,lables=[]):
    
    """
    "This function stacks the images for better view"
    
    parameters
    imgArray : array of images to be stacked(list of arrays expected)
    scale    : scaling factor of the images
    labels   : labels of the array of images

    """
    
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y],
                                            (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], 
                                                                                cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
    
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], 
                                     (0, 0), None, scale, scale)
            
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x],
                                                                       cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
#         print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),
                              (c*eachImgWidth+len(lables[d][c])*13+27,
                               30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],
                            (eachImgWidth*c+10,eachImgHeight*d+20),
                            cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0),1)
                
    return ver