# Augmented_reality
Used to display simple images of pokemon onto aruco cards.

Found on github: https://github.com/dwebb9/Augmented_reality

## Idea Generation
We considered doing a few different things including Y mountain, bathroom markers on a map, and a football field. We landed on pokemon due to it being fun, extensible (can do as many cards as you want), and different than what was mentioned in class.

## Challenges
The largest challenge we had was openCV struggling to find the markers. It turned out this was due to their not being a contrasting border around them - the red/blue of some cards was too close to black of the marker. We fixed this by adding a white border around the tags, and by adjusting the thresholding parameters in the aruco function to make it more likely to find them.

Another challenge, which we never overcame, was using proper shading on our objects. To do this, OpenGL has to be used, and we just didn't have enough experience with it to figure it out in time. Rather, we just filled each polygon of our object with a solid color shading. This gives good outlines, but lacks a depth that'd come from a more sophisticated approach.

Finally, it was also challenging to determine the 3D pose of the marker. Initially, we discoverd that we were actualy finding the homography of the markers which, although does work for plotting 2D imagaes on to the markers, does not work for 3D images.

## Solutions
We chose aruco markers to identify the pose of our cards since there is built in libraries in openCV for handling them. No need to re-invent the wheel!

OpenCV was used to detect the markers in our images, and then to estimate the pose of the markers to project our model onto the image. As mentioned above, we did this by filling in each of the polygons defined by the object as a single color.

We overcame the homography issue by using the function cv2.estimatePoseSingleMarkers() instead of cv2.findHomography(). 

## Results
Here's a video of all the cards: 

Here's a video verifying it's done live: 
