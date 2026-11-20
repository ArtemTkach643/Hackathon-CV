import numpy as np
import matplotlib 
import cv2

globalmap = cv2.imread("data/global_map.png")
test1 = cv2.imread("data/crops/crop_0.png")

gray1 = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
graymap = cv2.cvtColor(globalmap, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(graymap, None)

mapres = cv2.drawKeypoints(globalmap, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

result1 = cv2.drawKeypoints(test1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

bf = cv2.BFMatcher()

matches = bf.knnMatch(descriptors1, descriptors2, k=2)

good = []

print(len(matches))

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

print(len(good))

src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

print(int(mask.sum()))

h, w = gray1.shape
corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
box = cv2.perspectiveTransform(corners, H)

cv2.imwrite("results/mapres.png", mapres)
cv2.imwrite("results/result1.png", result1)

