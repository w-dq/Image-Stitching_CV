import cv2 
from matplotlib import pyplot as plt
import numpy as np

OUTPUT_FILE = 'output/'
INPUT_FILE = 'input/'

ransacReprojThreshold = 3
GOOD_MATCH_RATIO = 0.7
RESIZE_RATIO = 4

class Stitcher(object):
	"""docstring for Stitch"""
	def __init__(self, imageList):
		self.image_list = imageList
		self.image_number = len(self.image_list)
		self.homographies = [None] * self.image_number
		self.resize_image(RESIZE_RATIO)
		self.get_homography_pairs()

	def resize_image(self,resize_factor):
		if self.image_list:
			new_image = []
			for image in self.image_list:
				new_image.append(cv2.resize(image,(int(image.shape[1]/resize_factor),  \
												   int(image.shape[0]/resize_factor))))
			self.image_list = new_image
		else:
			print('empty image_list')


	def get_homography_pairs(self):

		kp_img_src, kp_src, des_src = self.sift_kp_des(self.image_list[0])
		kp_img_dst, kp_dst, des_dst = self.sift_kp_des(self.image_list[0])
		good_kp = self.get_good_match(des_src, des_dst)
		ptsA = np.float32([kp_src[m.queryIdx].pt for m in good_kp])
		H, status = cv2.findHomography(ptsA,ptsA,cv2.RANSAC,ransacReprojThreshold)
		self.homographies[0] = H

		for idx in range(0,self.image_number-1):
			src_idx = idx
			dst_idx = idx+1

			kp_img_src, kp_src, des_src = self.sift_kp_des(self.image_list[src_idx])
			kp_img_dst, kp_dst, des_dst = self.sift_kp_des(self.image_list[dst_idx])
			good_kp = self.get_good_match(des_src, des_dst)

			output = cv2.drawMatches(kp_img_src,kp_src,kp_img_dst,kp_dst,good_kp,None,flags=2)
			plt.imshow(output)
			plt.show()
			cv2.imwrite(OUTPUT_FILE + 'good_kp{}-{}.jpg'.format(src_idx,dst_idx), output)

			if len(good_kp) > 4:
				ptsA = np.float32([kp_dst[m.trainIdx].pt for m in good_kp])
				ptsB = np.float32([kp_src[m.queryIdx].pt for m in good_kp])
				H, status = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)
				self.homographies[dst_idx] = H

	def get_pair_stitch(self):
		for idx in range(1,self.image_number):
			stationary_img = self.image_list[idx-1]
			perspective_img = self.image_list[idx]
			perspective_output = cv2.warpPerspective(perspective_img, self.homographies[idx], \
											(stationary_img.shape[1]+perspective_img.shape[1], stationary_img.shape[0]))
			cv2.imwrite(OUTPUT_FILE + 'perspective{}.jpg'.format(idx), perspective_output)
			stationary_ouptput = cv2.warpPerspective(stationary_img, self.homographies[0], \
											(stationary_img.shape[1]+perspective_img.shape[1], stationary_img.shape[0]))
			result = self.blend_image(stationary_ouptput,perspective_output)
			cv2.imwrite(OUTPUT_FILE + 'pair{}-{}.jpg'.format(idx-1,idx), result)

	def get_total_stitch(self):
		perspective_list = self.image_list
		shape = [-1200,0]
		for i in self.image_list:
			shape[0] += i.shape[1]
			shape[1] = i.shape[0]
		final = np.float32([[[0]*3]*shape[0]]*shape[1])

		for idx_H,H in enumerate(self.homographies):
			for idx_i,image in enumerate(perspective_list):
				if idx_i >= idx_H:
					perspective_list[idx_i] = cv2.warpPerspective(image, H, \
							(shape[0], shape[1]))
		# now blend
		for i in perspective_list:
			final = self.blend_image(final,i)
		cv2.imwrite(OUTPUT_FILE + 'panaroma.jpg', final)
		# for idx in range(self.image_number):
		# 	for i in self.image_list:
		# 	stationary_img = self.image_list[idx-1]
		# 	perspective_img = self.image_list[idx]

	@staticmethod
	def blend_image(stationary,perspective):
		gray = cv2.cvtColor(perspective, cv2.COLOR_BGR2GRAY)
		for i in range(gray.shape[0]):
			for j in range(gray.shape[1]):
				if gray[i][j] != 0:
					stationary[i][j] = perspective[i][j]
		return stationary
		

	@staticmethod
	def sift_kp_des(img):
		sift = cv2.xfeatures2d.SIFT_create()
		kp, des = sift.detectAndCompute(img, None)
		kp_img = cv2.drawKeypoints(img, kp, None)
		return kp_img, kp, des

	@staticmethod
	def get_good_match(des1,des2):
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)
		flann = cv2.FlannBasedMatcher(index_params,search_params)
		matches = flann.knnMatch(des1,des2,k=2)

		good_kp = []
		for (i,j) in matches:
			if i.distance < GOOD_MATCH_RATIO * j.distance:
				good_kp.append(i)
		return good_kp

source_img_1 = cv2.imread(INPUT_FILE + '1.jpg', cv2.IMREAD_COLOR)
source_img_2 = cv2.imread(INPUT_FILE + '2.jpg', cv2.IMREAD_COLOR)
source_img_3 = cv2.imread(INPUT_FILE + '3.jpg', cv2.IMREAD_COLOR)
source_img_4 = cv2.imread(INPUT_FILE + '4.jpg', cv2.IMREAD_COLOR)


stitch = Stitcher([source_img_1,source_img_2,source_img_3,source_img_4])
stitch.get_pair_stitch()
stitch.get_total_stitch()







			
			
					





		
		
