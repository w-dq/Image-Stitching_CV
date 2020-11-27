import cv2 
from matplotlib import pyplot as plt
import numpy as np

IMG_ROOT = 'image/'

left_image = 'n.JPG'
right_image = 'm.JPG'

def sift_kp_des(img):
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img, None)
	kp_img = cv2.drawKeypoints(img, kp, None)
	return kp_img, kp, des

def get_good_match(des1,des2):

	# kdtree = cv2.ml.KNearest_create()
	# kdtree.setAlgorithmType(2)
	# print(kdtree.getAlgorithmType())
	# retval, results, neighborResponses, dist=kdtree.findNearest(des1, des2, k=2)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	# bf = cv2.BFMatcher()
	# matches = bf.knnMatch(des1, des2, k=2)

	good_kp = []
	for (i,j) in matches:
		if i.distance < 0.6 * j.distance:
			good_kp.append([i])
	return good_kp

def stichImage(img_1,img_2):
	kp_img_1, kp_1, des_1 = sift_kp_des(img_1)
	kp_img_2, kp_2, des_2 = sift_kp_des(img_2)
	good_kp = get_good_match(des_1, des_2)
	good_kp = sorted(good_kp,key = lambda x: x[0].distance)

	img3 = cv2.drawMatchesKnn(img_1,kp_1,img_2,kp_2,good_kp,None,flags=2)
	plt.imshow(img3)
	plt.show()

	if len(good_kp) > 4:
		ptsA = np.float32([kp_1[m[0].queryIdx].pt for m in good_kp])
		ptsB = np.float32([kp_2[m[0].trainIdx].pt for m in good_kp])
		ransacReprojThreshold = 3
		H, status = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)

		imgOutput = cv2.warpPerspective(img_1, H, (img_1.shape[1]+img_2.shape[1], img_1.shape[0]))

		H1, status = cv2.findHomography(ptsA,ptsA)
		stationary = cv2.warpPerspective(img_2, H1, (img_1.shape[1]+img_2.shape[1], img_1.shape[0]))
		cv2.imwrite('Stationary.jpg', stationary)
		cv2.imwrite('Perspective.jpg', imgOutput)

		imgOutput[0:img_2.shape[0],0:img_2.shape[1]] = img_2

	return imgOutput

source_img_1 = cv2.imread(IMG_ROOT + right_image, cv2.IMREAD_COLOR)
source_img_1 = cv2.resize(source_img_1,(int(len(source_img_1[0])/8),int(len(source_img_1)/8)))
source_img_2 = cv2.imread(IMG_ROOT + left_image, cv2.IMREAD_COLOR)
source_img_2 = cv2.resize(source_img_2,(int(len(source_img_2[0])/8),int(len(source_img_2)/8)))

output = stichImage(source_img_1,source_img_2)
cv2.imwrite('result.jpg', output)

