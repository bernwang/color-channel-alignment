# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform
import skimage.filters

# name of the input file
imname = 'data/village.tif'

# read in the image
im = skio.imread(imname)

print("dimension: ", im.shape)
# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)

# skio.imshow(im)
# skio.show()

# compute the height of each part (just 1/3 of total)
height = im.shape[0] // 3
width = im.shape[1]
print("height: ", height)
# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# def get_blockwise_gradient(m, blocksize=4):
# 	blockwise_gradient = np.zeros(m.shape)
# 	h, w = m.shape
# 	for j in range(w):
# 		lower_bound = max(0, j - blocksize)
# 		upper_bound = min(w, j + blocksize)
# 		for i in range(h):
# 			gradient = np.max(np.mean(m[i, lower_bound:j]) - np.mean(m[i, j:upper_bound+1]))
# 			blockwise_gradient[i,j] = gradient
# 	return blockwise_gradient

def SSD(im1, im2):
	try:
		return -np.linalg.norm(np.abs(np.gradient(im1)[0]) - np.abs(np.gradient(im2)[0])) - np.linalg.norm(np.abs(np.gradient(im1)[1]) - np.abs(np.gradient(im2)[1]))
	except ValueError:
		return -np.linalg.norm(im1 - im2)

def NCC(im1, im2):
	im1, im2 = im1.reshape((im1.size, )), im2.reshape((im2.size, ))
	im1_normalized, im2_normalized = im1 / np.linalg.norm(im1), im2 / np.linalg.norm(im2)
	# return im1_normalized.dot(im2_normalized)

	# try:
	# 	grad1, grad2 = np.gradient(im1), np.gradient(im2)
	# 	l = grad1[0].size
	# 	grad1x, grad2x = grad1[0].reshape((l, )), grad2[0].reshape((l, ))
	# 	grad1y, grad2y = grad1[1].reshape((l, )), grad2[1].reshape((l, ))

	# 	grad1x, grad2x = grad1x / np.linalg.norm(grad1x), grad2x / np.linalg.norm(grad2x)
	# 	grad1y, grad2y = grad1y / np.linalg.norm(grad1y), grad2y / np.linalg.norm(grad2y)
	# 	return grad1x.dot(grad2x) + grad1y.dot(grad2y)
	# except ValueError:
	return im1_normalized.dot(im2_normalized)

def pyramid_find_displacement(im, fixed, n, e, matching_metric=SSD):
	if n == 0:
		return find_displacement(im, fixed)
	im_resized = sk.transform.rescale(sk.filters.gaussian(im), .5)
	fixed_resized = sk.transform.rescale(sk.filters.gaussian(fixed), .5)
	displacement_vector = pyramid_find_displacement(im_resized, fixed_resized, n - 1, e + 2) * 2
	print("displacement vector: ", displacement_vector)
	return find_neighboring_displacement(im, fixed, displacement_vector, e, matching_metric)

def find_neighboring_displacement(im, fixed, displacement_vector, e, matching_metric=SSD):
	copy = np.copy(im)
	best_displacement_x = displacement_vector[0]
	best_displacement_y = displacement_vector[1]
	best_matching_score = float("-inf")
	h, w = im.shape
	x_lower_bound = max(best_displacement_x - e, -h//10 + 1)
	x_upper_bound = min(best_displacement_x + e, h//10 - 1) + 1
	y_lower_bound = max(best_displacement_y - e, -w//10 + 1)
	y_upper_bound = min(best_displacement_y + e, w//10 - 1) + 1
	for i in range(x_lower_bound, x_upper_bound):
		for j in range(y_lower_bound, y_upper_bound):
			im = np.roll(np.roll(copy, j, axis=1), i, axis=0)
			score = matching_metric(fixed, im)
			if score > best_matching_score:
				best_displacement_x, best_displacement_y, best_matching_score = i, j, score
			# print("\ti:{}, j, {}, x: [{}, {}], {}; y: [{},  {}], {}, score: {}, best score: {}".format(i, j, x_lower_bound, x_upper_bound, best_displacement_x, y_lower_bound, y_upper_bound, best_displacement_y, score, best_matching_score))

	return np.array([best_displacement_x, best_displacement_y])


def find_displacement(im, fixed, matching_metric=SSD):
	copy = np.copy(im)
	best_displacement_x = 0
	best_displacement_y = 0
	best_matching_score = float("-inf")
	h, w = im.shape
	for i in range(-h + 1, h):
		for j in range(-w + 1, w):
			im = np.roll(np.roll(copy, j, axis=1), i, axis=0)
			score = matching_metric(fixed, im)
			if score > best_matching_score:
				# print("\t{}, {}, score: {}".format(i, j, score))
				best_displacement_x, best_displacement_y, best_matching_score = i, j, score
	return np.array([best_displacement_x, best_displacement_y])

def pyramid_align(im, fixed, matching_metric=SSD):
	# skio.imshow_collection(np.gradient(im))
	# print(np.mean(np.gradient(im)[0]))
	# skio.show()
	n = int(np.floor(np.log2(np.min(im.shape))))
	displacement_x, displacement_y = pyramid_find_displacement(im, fixed, n, 1, matching_metric)
	print(displacement_x, displacement_y)
	return np.roll(np.roll(im, displacement_y, axis=1), displacement_x, axis=0)

def crop(r, g, b):
	for i in range(4):
		r, g, b = crop_one_side(r, g, b)
		r, g, b = np.rot90(r), np.rot90(g), np.rot90(b)
	return r, g, b

def crop_one_side(r, g, b):
	# r_gradient, g_gradient, b_gradient = get_blockwise_gradient(r), get_blockwise_gradient(g), get_blockwise_gradient(b)
	# r_gradient, g_gradient, b_gradient = np.gradient(r)[0], np.gradient(g)[0], np.gradient(b)[0]
	# r_gradient, g_gradient, b_gradient = difference_of_gaussians(r), difference_of_gaussians(g), difference_of_gaussians(b)
	r_gradient, g_gradient, b_gradient = r, g, b
	# print(r_gradient[:50,50])
	# print(np.mean(r_gradient[:,50]))
	# print(np.max(np.abs(b_gradient[:,5])))
	row_to_crop = max(find_row_to_crop(r_gradient), find_row_to_crop(g_gradient), find_row_to_crop(b_gradient))
	print("row to crop: ", row_to_crop)
	r_cropped, g_cropped, b_cropped = r[:,row_to_crop:], g[:,row_to_crop:], b[:,row_to_crop:]

	row_to_crop = max(find_row_to_crop_black(r_cropped), find_row_to_crop_black(g_cropped), find_row_to_crop_black(b_cropped))
	print("row to crop: ", row_to_crop)
	r_cropped, g_cropped, b_cropped = r_cropped[:,row_to_crop:], g_cropped[:,row_to_crop:], b_cropped[:,row_to_crop:]

	return r_cropped, g_cropped, b_cropped

def find_row_to_crop(im, thresh=.90):
	start = 0
	h, w = im.shape
	end = w
	m = 8 #(start + end) // 2
	blocksize = 8
	# while (m < w and np.mean(np.abs(im[:,m])) < thresh):
	while (m < w // 10):
		if np.mean(np.abs(im[:,m:m+blocksize])) < thresh:
			if blocksize == 1:
				break
			else:
				blocksize -= 1
		else:
			m += blocksize
		# print(np.mean(np.abs(im[:,m:m+blocksize])))
	return m

def find_row_to_crop_black(im, thresh=.2):
	start = 0
	h, w = im.shape
	end = w
	m = 0 #(start + end) // 2
	blocksize = 8
	while (m < w // 10):
		if np.mean(np.abs(im[:,m:m+8])) > thresh:
			if blocksize == 1:
				break
			else:
				blocksize -= 1
		else:
			m += blocksize
		print(m, np.mean(np.abs(im[:,m:m+blocksize])))
		m += blocksize
	return m


def difference_of_gaussians(im, num_differences=5):
	sigmas = np.linspace(0, 2, 2 * num_differences)
	difference = np.zeros(im.shape)
	for i in range(num_differences):
		sigma1 = sigmas[2*i]
		sigma2 = sigmas[2*i+1]
		difference += sk.filters.gaussian(im, sigma=sigma1) - sk.filters.gaussian(im, sigma=sigma2)
	return difference / num_differences * 100
# def align(im, fixed, matching_metric=NCC):
# 	# im = np.roll(im, -displacement_bound, axis=0)
# 	# im = np.roll(im, -displacement_bound, axis=1)
# 	copy = np.copy(im)
# 	best_displacement_x = 0
# 	best_displacement_y = 0
# 	if np.max(im.shape) < 2000:
# 		image_pryamid_scale = np.array([0])
# 		sigmas = np.array([0])
# 	else:
# 		image_pyramid_scale = np.arange(int(np.floor(np.log2(np.min(im.shape)))), -1, -1)
# 		sigmas = np.linspace(2, 0, len(image_pyramid_scale))
# 	print("image pyramid scales: ", image_pyramid_scale)
# 	for k in range(len(image_pyramid_scale)):
# 		best_matching_score = float("-inf")
# 		scaling_factor = 2**image_pyramid_scale[k]
# 		print("scaling_factor: ", scaling_factor)
# 		copy_rescaled = sk.transform.rescale(sk.filters.gaussian(copy, sigma=sigmas[k]), 1 / scaling_factor)
# 		# skio.imshow(copy_rescaled)
# 		# skio.show()
# 		fixed_rescaled = sk.transform.rescale(sk.filters.gaussian(fixed, sigma=sigmas[k]), 1 / scaling_factor)
# 		print("\tnew size: ", copy_rescaled.shape)
# 		lower_bound_x = max(best_displacement_x - scaling_factor, -copy_rescaled.shape[0] // 2 + 1)
# 		upper_bound_x = min(best_displacement_x + scaling_factor, copy_rescaled.shape[0] // 2 - 1) + 1
# 		lower_bound_y = max(best_displacement_y - scaling_factor, -copy_rescaled.shape[1] // 2 + 1)
# 		upper_bound_y = min(best_displacement_y + scaling_factor, copy_rescaled.shape[1] // 2 - 1) + 1
# 		print("\tx: [{}, {}], {}; y: [{}, {}], {}".format(lower_bound_x, upper_bound_x, best_displacement_x, lower_bound_y, upper_bound_y, best_displacement_y))
# 		for i in range(lower_bound_x, upper_bound_x):
# 			for j in range(lower_bound_y, upper_bound_y):
# 				im = np.roll(np.roll(copy_rescaled, j, axis=1), i, axis=0)
# 				score = matching_metric(fixed_rescaled, im)
# 				# print("\tscore: ", score)
# 				if score > best_matching_score:
# 					print("\t{}, {}, score: {}".format(i, j, score))
# 					best_displacement_x, best_displacement_y, best_matching_score = i * 2, j * 2, score
# 	print(best_displacement_x // 2, best_displacement_y // 2)
# 	return np.roll(np.roll(copy, best_displacement_y // 2, axis=1), best_displacement_x // 2, axis=0)

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
# ag = g
# ar = r

r_cropped, g_cropped, b_cropped = crop(r, g, b)

ar = pyramid_align(r_cropped, b_cropped)
ag = pyramid_align(g_cropped, b_cropped)
# create a color image
im_stacked = np.dstack([r, g, b])
im_out = np.dstack([ar, ag, b_cropped])
im_manual = np.dstack([np.roll(np.roll(r, 120, axis=0), 8, axis=1), np.roll(np.roll(g, 50, axis=0), 30, axis=1), b])
# save the image
fname = '/out_path/out_fname.jpg'
# skio.imsave(fname, im_out)
print("cropped size: ", r_cropped.shape)
# display the image
# skio.imshow(im_out)
skio.imshow_collection([ar, ag, b_cropped])
# skio.imshow_collection([r_cropped, g_cropped, b_cropped, im_out])
skio.show()
