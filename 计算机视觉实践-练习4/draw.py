import cv2
import matplotlib.pyplot as plt

# 读取两张图片
img1 = cv2.imread('img1.jpg')  # queryImage
img2 = cv2.imread('img2.jpg')  # trainImage

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 找到两张图片中的关键点和描述符
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

matched_image_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

plt.imshow(matched_image_rgb),plt.show()
