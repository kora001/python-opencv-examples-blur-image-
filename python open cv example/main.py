import cv2
import numpy as np

# Read the image file
img = cv2.imread("yesil_sapka.jpg")

# Apply median blur to the image
blur_image = cv2.medianBlur(img, 9)

# Convert the image from BGR color space to HSV color space
dataHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the color range to be included in the mask
lower_color = np.array([0, 42, 0])
upper_color = np.array([179, 255, 255])

# Create a binary mask using the color range defined above
mask = cv2.inRange(dataHsv, lower_color, upper_color)

# Dilate the binary mask to fill gaps between contours
mask = cv2.dilate(mask, (3, 3), iterations=3)

# Find contours on the binary mask
contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
for i in range(len(contour)):
    cv2.drawContours(img, contour, i, (0, 0, 255), 4)

# Display the original image, HSV image, binary mask, and blurred image
cv2.imshow("Original", img)
cv2.imshow("Hsv", dataHsv)
cv2.imshow("Mask", mask)
cv2.imshow("blurred_image", blur_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
