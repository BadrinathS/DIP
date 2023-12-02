import numpy as np
import cv2
from scipy import signal
import os
import matplotlib.pyplot as plt



def gaussian(x,sigma):
    coef = (1.0/np.sqrt((2*np.pi*(sigma**2))))

    return coef*np.exp(-(x**2)/(2*(sigma**2)))

def distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def gaussian_smoothing(img = cv2.imread('./Images/building_noisy.png', cv2.IMREAD_GRAYSCALE).astype(float), sigma=100):
    
    # g_kernel = np.zeros((7,7))
    # for i in range(7):
    #     for j in range(7):
    #         g_kernel[i,j] = gaussian(distance(i,j,3,3), sigma)
    
    # g_kernel /= np.sum(g_kernel)

    g_kernel = gaussian_kernel(7, sigma)

    new_img = signal.convolve2d(img, g_kernel)
    cv2.imwrite('./gaussian_smoothing.png', new_img)

    return new_img


#function to create kernel of gaussian 
def gaussian_kernel(size, std_dev):
    """
    Function to generate the gaussian kernel
    """
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = (1 / (2 * np.pi * std_dev ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * std_dev ** 2))
    return kernel/np.sum(kernel) 



def get_neighbours(i,j, ker_size):
    
    neighbours = []
    for x in range(i-ker_size//2,i+ker_size//2+1):
        for y in range(j-ker_size//2, j+ker_size//2+1):
            neighbours.append([x,y])
    neighbours = np.array(neighbours)
    return neighbours


def bilateral_filter(image, ker_size, std_dev_lum, std_dev_sp):
    #store the image shape 
    M,N = image.shape

    #pad the image with zeros
    pad_width = ((ker_size//2, ker_size//2), (ker_size//2, ker_size//2))
    image = np.pad(image, pad_width, mode='constant', constant_values=0)
    
    # Create a blank image to store the filtered result
    filtered_image = np.zeros_like(image)
    
    #get the kernel of gaussian 
    spatial_weights = gaussian_kernel(ker_size, std_dev_sp)
    #iterate over all pixels
    for i in range(ker_size//2, M):
        for j in range(ker_size//2, N):
            kernel_indices = get_neighbours(i,j, ker_size)
            image_indices = image[kernel_indices[:,0],kernel_indices[:,1]]
            image_kernel = image_indices.reshape((ker_size, ker_size))
            image_kernel = image_kernel.astype(float)
            intensity_weights = np.exp(-((image_kernel[1,1] - image_kernel)**2)/(2*std_dev_lum**2))
            bi_image = image_kernel*intensity_weights*spatial_weights
            weights = intensity_weights*spatial_weights
            bi_image_ij = np.sum(bi_image)
            w_ij = np.sum(weights)
            filtered_image[i,j] = bi_image_ij/w_ij
    
    cv2.imwrite('./bilateral_result.png', filtered_image)
    return filtered_image
    

def laplace(img_orig, img_bilateral, img_gaussian):
    
    filter = np.zeros((3,3))
    
    filter[1,1] = -4
    filter[0,1] = 1
    filter[2,1] = 1
    filter[1,0] = 1
    filter[1,2] = 1
    
    new_img_orig = np.zeros(img_orig.shape).astype(np.int32)
    new_img_bilateral = np.zeros(img_bilateral.shape).astype(np.int32)
    new_img_gaussian = np.zeros(img_gaussian.shape).astype(np.int32)

    for r in range(img_orig.shape[0]):
        for c in range(img_orig.shape[1]):

            # coef = 0
            wp_orig = 0
            wp_bilateral = 0
            wp_gaussian = 0

            for k_r in range(filter.shape[0]):
                for k_c in range(filter.shape[1]):

                    conv_x = r - (int(filter.shape[0]/2) - k_r)
                    conv_y = c - (int(filter.shape[1]/2) - k_c)

                    if conv_x<0 or conv_y<0 or conv_x>=img_orig.shape[0] or conv_y>= img_orig.shape[1]:
                        
                        gaussian_distance = 0
                    else:
                        # coef += filter[k_r, k_c]
                        wp_orig += img_orig[conv_x, conv_y]*filter[k_r, k_c]
                        wp_bilateral += img_orig[conv_x, conv_y]*filter[k_r, k_c]
                        wp_gaussian += img_orig[conv_x, conv_y]*filter[k_r, k_c]
            # wp /= coef
            new_img_orig[r,c] = wp_orig
            new_img_bilateral[r,c] = wp_bilateral
            new_img_gaussian[r,c] = wp_gaussian




    cv2.imwrite('./laplacian_result_orig.png', new_img_orig)
    cv2.imwrite('./laplacian_result_bilateral.png', new_img_bilateral)
    cv2.imwrite('./laplacian_result_gaussian.png', new_img_gaussian)

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(img, cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(new_img, cmap='gray')
    # plt.show()

    return new_img_orig, new_img_bilateral, new_img_gaussian


def grad_oneaxis(img, M):

    new_img = np.zeros(img.shape).astype(np.int32)

    for r in range(img.shape[0]):
        for c in range(img.shape[1]):

            wp = 0
            for k_r in range(M.shape[0]):
                for k_c in range(M.shape[1]):

                    conv_x = r - (int(M.shape[0]/2) - k_r)
                    conv_y = c - (int(M.shape[1]/2) - k_c)

                    if conv_x<0 or conv_y<0 or conv_x>=img.shape[0] or conv_y>= img.shape[1]:
                        
                        gaussian_distance = 0
                    else:
                        wp += img[conv_x, conv_y]*M[k_r, k_c]
            new_img[r,c] = wp
    
    return new_img


def HoughTransform(img = cv2.imread('./Images/random_shapes.png', cv2.IMREAD_GRAYSCALE), name='test.png'):
    Mx = np.array([[-1, 0, 1], [-2,0,2], [-1,0,1]])
    My = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


    grad_x_img = signal.convolve2d(img, Mx)
    grad_y_img = signal.convolve2d(img, My)

    grad_mag = np.sqrt(grad_x_img**2 + grad_y_img**2)

    edge_map = np.where(grad_mag> 0.5*grad_mag.max(), 1, 0)

    diag_length = int(np.sqrt(np.sum(edge_map.shape[0]**2 +edge_map.shape[1]**2)))

    theta_range = np.linspace(-np.pi, np.pi, num = 180)
    dist_range = np.linspace(0, 2*diag_length, num=int(diag_length))

    cos_theta = np.cos(theta_range)
    sin_theta = np.sin(theta_range)

    accumulator = np.zeros((len(dist_range), len(theta_range)))

    y_ids, x_ids = np.nonzero(edge_map)

    for i in range(len(y_ids)):
        x = x_ids[i]
        y = y_ids[i]

        for t_id in range(len(theta_range)):
            rho = x*cos_theta[t_id] + y*sin_theta[t_id] + diag_length
            rho_index = np.argmin(np.abs(dist_range - int(rho)))
            accumulator[rho_index, t_id] += 1

    print(accumulator.max(), accumulator.min(), accumulator.mean())
    # accumulator_updated = accumulator

    cv2.imwrite('./accumulator.png', 255*accumulator/accumulator.max())

    accumulator_updated = np.where(accumulator>10*accumulator.mean(), 1, 0)

    rho_ids, t_ids = np.nonzero(accumulator_updated)

    new_edge_map = np.zeros(edge_map.shape)


    for i in range(len(rho_ids)):
        rho_val, theta = dist_range[rho_ids[i]] - diag_length, theta_range[t_ids[i]]

        for j in range(new_edge_map.shape[1]):

            y = (rho_val -j*np.cos(theta))/np.sin(theta)

            if y>= 0 and y< new_edge_map.shape[0]:
                new_edge_map[int(y),j] += 1

    # for j in range(new_edge_map.shape[1]):

    #         y = (100 -j*1/np.sqrt(2))/(1/np.sqrt(2))

    #         if y>= 0 and y< new_edge_map.shape[0]:
    #             new_edge_map[int(y),j] += 1

    new_edge_map = 255* (new_edge_map - new_edge_map.min())/(new_edge_map.max() - new_edge_map.min())

    cv2.imwrite(name, new_edge_map)
            







    

#####################   Problem 1   ####################
img = cv2.imread('./Images/building_noisy.png', cv2.IMREAD_GRAYSCALE).astype(float)

bilat = bilateral_filter(img,7,  50,100)

gauss = gaussian_smoothing(img, sigma=10)

laplace_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

image_lap = signal.convolve2d(img.astype(float), laplace_kernel)
bilat_lap = signal.convolve2d(bilat.astype(float), laplace_kernel)
gauss_lap = signal.convolve2d(gauss.astype(float), laplace_kernel)


# image_lap = 255* (image_lap - image_lap.min())/(image_lap.max() - image_lap.min())
# bilat_lap = 255* (bilat_lap - bilat_lap.min())/(bilat_lap.max() - bilat_lap.min())
# gauss_lap = 255* (gauss_lap - gauss_lap.min())/(gauss_lap.max() - gauss_lap.min())

cv2.imwrite('./laplacian_result_orig.png', image_lap)
cv2.imwrite('./laplacian_result_bilateral.png', bilat_lap)
cv2.imwrite('./laplacian_result_gaussian.png', 2*gauss_lap)



# loo = cv2.imread('laplacian_result_orig.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)
# log = cv2.imread('laplacian_result_gaussian.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)
# lob = cv2.imread('laplacian_result_bilateral.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)

# d_og = loo - log
# d_ob = loo - lob
# d_bg = lob - log


# print(loo.min(), loo.max())
# print(log.min(), log.max())
# print(lob.min(), lob.max())

# print(loo[27,99:109], log[27,99:109])

# print(d_og.min(), d_og.max())
# print(d_ob.min(), d_ob.max())
# print(d_bg.min(), d_bg.max())

# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(loo-log)
# plt.subplot(1,3,2)
# plt.imshow(log-lob)
# plt.subplot(1,3,3)
# plt.imshow(loo - lob)
# plt.show()


###################### end of problem 1 ##################


###################### Problem 2 ###########################

# filter = np.zeros((7,7))
    
# for i in range(filter.shape[0]):
#     for j in range(filter.shape[1]):
#         filter[i,j] = gaussian(np.sqrt((i - int(filter.shape[0]/2))**2 + (j - int(filter.shape[1]/2))**2), sigma=100)


# img1 = cv2.imread('Images/book_noisy1.png', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('Images/book_noisy2.png', cv2.IMREAD_GRAYSCALE)
# img3 = cv2.imread('Images/architecture_noisy1.png', cv2.IMREAD_GRAYSCALE)
# img4 = cv2.imread('Images/architecture_noisy2.png', cv2.IMREAD_GRAYSCALE)

# img1_gauss = signal.convolve2d(img1, filter)
# img2_gauss = signal.convolve2d(img2, filter)
# img3_gauss = signal.convolve2d(img3, filter)
# img4_gauss = signal.convolve2d(img4, filter)

# img1_gauss = gaussian_smoothing(img = cv2.imread('Images/book_noisy1.png', cv2.IMREAD_GRAYSCALE))
# img2_gauss = gaussian_smoothing(img = cv2.imread('Images/book_noisy2.png', cv2.IMREAD_GRAYSCALE))

# Mx = np.array([[-1, 0, 1], [-2,0,2], [-1,0,1]])
# My = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# grad_x_img1 = signal.convolve2d(img1_gauss, Mx)
# grad_y_img1 = signal.convolve2d(img1_gauss, My)

# # grad_x_img1 = grad_oneaxis(img1_gauss, Mx)
# # grad_y_img1 = grad_oneaxis(img1_gauss, My)

# grad_mag1 = np.sqrt(grad_x_img1**2 + grad_y_img1**2)
# grad_mag1 = 255*(grad_mag1 - grad_mag1.min())/(grad_mag1.max() - grad_mag1.min())

# cv2.imwrite('./grad_x_img1.png', grad_x_img1)
# cv2.imwrite('./grad_y_img1.png', grad_y_img1)
# cv2.imwrite('./grad_mag_img1.png', grad_mag1)


# grad_x_img2 = signal.convolve2d(img2_gauss, Mx)
# grad_y_img2 = signal.convolve2d(img2_gauss, My)

# # grad_x_img2 = grad_oneaxis(img2_gauss, Mx)
# # grad_y_img2 = grad_oneaxis(img2_gauss, My)


# grad_mag2 = np.sqrt(grad_x_img2**2 + grad_y_img2**2)
# # grad_mag2 = 255*(grad_mag2 - grad_mag2.min())/(grad_mag2.max() - grad_mag2.min())

# cv2.imwrite('./grad_x_img2.png', grad_x_img2)
# cv2.imwrite('./grad_y_img2.png', grad_y_img2)
# cv2.imwrite('./grad_mag_img2.png', grad_mag2)

# grad_x_img3 = signal.convolve2d(img3_gauss, Mx)
# grad_y_img3 = signal.convolve2d(img3_gauss, My)

# # grad_x_img2 = grad_oneaxis(img2_gauss, Mx)
# # grad_y_img2 = grad_oneaxis(img2_gauss, My)


# grad_mag3 = np.sqrt(grad_x_img3**2 + grad_y_img3**2)
# # grad_mag3 = 255*(grad_mag3 - grad_mag3.min())/(grad_mag3.max() - grad_mag3.min())

# cv2.imwrite('./grad_x_img3.png', grad_x_img3)
# cv2.imwrite('./grad_y_img3.png', grad_y_img3)
# cv2.imwrite('./grad_mag_img3.png', grad_mag3)


# grad_x_img4 = signal.convolve2d(img4_gauss, Mx)
# grad_y_img4 = signal.convolve2d(img4_gauss, My)

# # grad_x_img2 = grad_oneaxis(img2_gauss, Mx)
# # grad_y_img2 = grad_oneaxis(img2_gauss, My)


# grad_mag4 = np.sqrt(grad_x_img4**2 + grad_y_img4**2)
# # grad_mag4 = 255*(grad_mag4 - grad_mag4.min())/(grad_mag4.max() - grad_mag4.min())

# cv2.imwrite('./grad_x_img4.png', grad_x_img4)
# cv2.imwrite('./grad_y_img4.png', grad_y_img4)
# cv2.imwrite('./grad_mag_img4.png', grad_mag4)

# print(grad_mag1.min(), grad_mag1.max(), grad_mag1.mean())
# print(grad_mag2.min(), grad_mag2.max(), grad_mag2.mean())
# print(grad_mag3.min(), grad_mag3.max(), grad_mag3.mean())
# print(grad_mag4.min(), grad_mag4.max(), grad_mag4.mean())

# # threshold1 = 0.5* grad_mag1.max()
# # threshold2 = 0.5* grad_mag2.max()
# # threshold3 = 0.5* grad_mag3.max()
# # threshold4 = 0.5* grad_mag4.max()

# threshold1 = 25
# threshold2 = 70
# threshold3 = 25
# threshold4 = 25



# edge1 = np.where(grad_mag1>threshold1, 255, 0)
# edge2 = np.where(grad_mag2>threshold2, 255, 0)
# edge3 = np.where(grad_mag3>threshold3, 255, 0)
# edge4 = np.where(grad_mag4>threshold4, 255, 0)

# cv2.imwrite('./img1_threshold_025.png', edge1)
# cv2.imwrite('./img2_threshold_50.png', edge2)
# cv2.imwrite('./img3_threshold_025.png', edge3)
# cv2.imwrite('./img4_threshold_025.png', edge4)

# plt.figure()
# plt.subplot(2,2,1)
# plt.imshow(edge1)
# plt.subplot(2,2,2)
# plt.imshow(edge2)
# plt.subplot(2,2,3)
# plt.imshow(edge3)
# plt.subplot(2,2,4)
# plt.imshow(edge4)
# plt.show()

####################### End of Problem 2 ########################


####################### Problem 3 ########################

# HoughTransform(img = cv2.imread('./Images/random_shapes.png', cv2.IMREAD_GRAYSCALE), name='hough_random_shapes.png')
# HoughTransform(img = cv2.imread('./Images/random_shapes_noisy.png', cv2.IMREAD_GRAYSCALE), name='hough_random_shapes_noisy.png')
# HoughTransform(img = cv2.imread('./Images/random_shapes_occlusion.png', cv2.IMREAD_GRAYSCALE), name='hough_random_shapes_occlusion.png')
# HoughTransform(img = cv2.imread('./Images/building_noisy.png', cv2.IMREAD_GRAYSCALE), name='hough_building_noisy.png')


####################### Enf of Problem 3 ########################