''' This python file uses the Santa image for Augment Reality. In particular this
code detects the Santa image when placed in front of the web cam, compute a set of
correspondence and homography and then overlay the grinch.jpg 

Step 1: Feature Matching & Homography
i. SIFT features in images and apply the ratio test to find the best matches.
(As sift.create() didn't work fro me I have used Kaze detector to detect features.)
ii. Now we set a condition that atleast 30 matches (defined by MIN_MATCH_COUNT) are to be there to find the object. 
Otherwise simply show a message saying not enough matches are present.
iii. If enough matches are found, we extract the locations of matched keypoints in both the images.

Step 2: Compute Homography
i. Matches are passed to find the perpective transformation. Once we get this 3x3 transformation matrix, 
we use it to transform the corners of queryImage to corresponding points in targetImage. 
Then we draw it.
ii. Using the resultant homography to project a outline onto the target in the query image.
iii. A second inner box is drawn so as to place the Grinch image within box as well as to observe Santa image in the output.
iv. A flag variable is set to true at this stage to identify Santa image is detected. 

Step 3 : Warpperspective
Using the warpprespective function and an image of choice, project the image such that it 
appears to lie in the plane of Santa.

Note:
The image that needs to be overlapped needs to be saaved with the name Grinch.jpg
The overlayed image is displayed another window and within the inner box as as to enable the user to see Santa image in bavckground.

Deepthi Jain Brahmadev
Student ID: 19252262
'''

import cv2
import numpy as np
import glob

#Opens the web cam 
cap = cv2.VideoCapture(0)
target = cv2.imread('santa.jpg')

#A condition is set such that atleast 30 matches are to be there to find the object. 
#Otherwise show a message saying not enough matches are present.
MIN_MATCH_COUNT = 30



#Construct a kaze object.
#Different parameters can be passed to it which are optional. 
feature_detector = cv2.KAZE_create()

#This method detects keypoints and computes the descriptors.
#Descriptors describe elementary characteristics such as the shape, the color, the texture or the motion, among others.
(tkp,tdes) = feature_detector.detectAndCompute(target,None)
FLANN_INDEX_KDTREE = 0

#FLANN_INDEX_KDTREE algorithm is used.
#Dictionary (key-value) pair is used to search parameters.
#For FLANN based matcher, we need to pass two dictionaries which specifies the algorithm to be used. 
#First one is IndexParams.
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

# It specifies the number of times the trees in the index should be recursively traversed. 
# Higher values gives better precision, but also takes more time.
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


#creates a window that is used as a placeholder for image.
cv2.namedWindow("Matches",cv2.WINDOW_AUTOSIZE)


while (True):
    #reads data from video capture.
    ret, query = cap.read()

    #resize of the live image fitting it to the frame size as that of target image.
    frame_width, frame_height, frame_depth = query.shape

    #This method detects keypoints and computes the descriptors.
    #Descriptors describe elementary characteristics such as the shape, the color, the texture or the motion, among others.
    #Query image is the live image that is captured from webcam.
    (qkp,qdes) = feature_detector.detectAndCompute(query,None)

    #match() method to get the best matches in two images.
    matches = flann.knnMatch(tdes,qdes,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
   
    #matches are sorted in ascending order of their distances so that best matches (with low distance) come to front. 
    for m,n in matches:
        #if distance i sless then descriptor is appended to array good
        if m.distance < 0.7*n.distance:
            good.append(m)
    #length of good array is compared with min match count defined at the start.
    #If it is greater then array is reshaped.  
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ tkp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ qkp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = target.shape
        #Draws inner rectangle to place Grinch.jpg within the box
        #This step can be omitted if th euser doesn't want to see santa image in the background.
        h2 = h-100
        w2 = w-100
        d2 = d-100
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        query = cv2.polylines(query,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        pts2 = np.float32([ [0,0],[0,h2-1],[w2-1,h2-1],[w2-1,0] ]).reshape(-1,1,2)
        dst2 = cv2.perspectiveTransform(pts2,M)
        query = cv2.polylines(query,[np.int32(dst2)],True,255,3, cv2.LINE_AA)

        #This variable is set so as to perform warpperspective only if Santa image is deetcted.
        flag = True
        
        if flag == True:
        
            maskThreshold=10
            img = cv2.imread('grinch.jpg')
            img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite('grinch1.jpg', img_rotate_90_clockwise)
            # Webcam capture
            cap = cv2.VideoCapture(0)
            images = glob.glob('grinch1.JPG')  # The .jpg images in the folder can be displayed
            currentImage = 0  # the first image is selected
            replaceImg = cv2.imread(images[currentImage])
            rows, cols, ch = replaceImg.shape
            pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])  # this points are necessary for the transformation

            # compute the transform matrix
            M = cv2.getPerspectiveTransform(pts1, dst2)
            rows, cols, ch = query.shape  
            # make the perspective change in a image of the size of the camera input
            dst = cv2.warpPerspective(replaceImg, M, (cols, rows))
            # A mask is created for adding the two images
            # maskThreshold is a variable because that allows to subtract the black background from different images
            ret, mask = cv2.threshold(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), maskThreshold, 1, cv2.THRESH_BINARY_INV)
            # Erode and dilate are used to delete the noise
            mask = cv2.erode(mask, (3, 3))
            mask = cv2.dilate(mask, (3, 3))
            for c in range(0, 3):
                query[:, :, c] = dst[:, :, c] * (1 - mask[:, :]) + query[:, :, c] * mask[:, :]


        # Finally the result is displayed in another window.
            cv2.imshow('img', query)

    #If len of good matches is < min match count defined then no enough matches is printed.
    else:
        print ("Not enough matches: %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    
    draw_params = dict(matchColor = (0,255,0), 
    singlePointColor = None,
    matchesMask = matchesMask, # draw only inliers
    flags = 2)

    # This method draws matches of keypoints from two images in the output image.
    # Match is a line connecting two keypoints (circles).
    corr_img = cv2.drawMatches(target,tkp,query,qkp,good,None,**draw_params)
    corr_img = cv2.resize(corr_img, (0,0), fx=0.5, fy=0.5)
    cv2.imshow("Matches",corr_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


'''
Source:
    1. https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
    2. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
'''
