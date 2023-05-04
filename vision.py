import cv2
import numpy as np
def bb_intersection_over_union(boxA, boxB):
	# get the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])

	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = ( (xB - xA )+ 1) * ( (yB - yA) + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


# images scale
target_size = (500, 500)

# input image
img = cv2.imread('D:/train/10.png')

# scaling image
resized = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

# blurring the resized image using gaussian
blurred = cv2.GaussianBlur(resized, (5, 5), 0)

# bgr to greyscale
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

# putting a thresshlod
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area
max_area = 0
max_contour = None
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > max_area:
        max_area = area
        max_contour = cnt



# test image box that we compare with
gt_box = [25,6,9,27]  # x, y, w, h

# bounding box coordinates
x, y, w, h = cv2.boundingRect(max_contour)

# scaling box back to original image size
x *= 2
y *= 2
w *= 2
h *= 2

# drawing rectangle
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Calculate the IOU
contour_box = [x, y, w, h]
iou = bb_intersection_over_union(contour_box, gt_box)
print("IOU:", iou)




# Resize the image to the target size
scaled = cv2.resize(img, target_size)

# Display the image with the rectangle around the number
cv2.imshow('image', scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()