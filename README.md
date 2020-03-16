# SIFT
This python file uses the Santa image for Augment Reality. In particular this
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
deepthinithin1920@gmail.com
