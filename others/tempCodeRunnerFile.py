import cv2

# Load two images
img1 = cv2.imread('totoro.jpg', 0)
img2 = cv2.imread('castle.jpg', 0)

# Detect ORB features
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute-Force Matcher [6]
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Draw top 25 matches
matches = sorted(matches, key=lambda x:x.distance)
result = cv2.resize(cv2.drawMatches(img1, kp1, img2, kp2, matches[:25], None), (1600, 800))

cv2.imshow('Feature Matches', result)
cv2.waitKey(0)
