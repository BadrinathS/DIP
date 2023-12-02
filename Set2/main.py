import matplotlib.pyplot as plt
from skimage.io import imread
import cv2
import os
import numpy as np


def Question1():
    im = imread('./Images/ECE.png', as_gray=False)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.show()


def histogram_equalization(img_path='./Images/hazy.png'):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_copy = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hist = [0 for i in range(256)]
    # hist_copy = [0 for i in range(256)]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i,j]] += 1
    
    # plt.figure()
    # plt.title('Histogram of image')
    # plt.subplot(1,2,1)
    # plt.title('Before equalization')
    # plt.bar(np.arange(len(hist)),hist)

    for j in range(1,len(hist)):
        hist[j] = hist[j] + hist[j-1]
    
    # plt.figure()
    # plt.bar(np.arange(len(hist)), hist)
    # plt.savefig('./results/CDF_histeq.png')
    # plt.show()
    # exit()

    hist = [int(255*j/hist[-1]) for j in hist]
    # for i,val in enumerate(hist):
    #     hist_copy[val] += hist[i]
    
    for i in range(len(hist)):
        hist[i] = (hist[i] - min(hist))*(255-0)/(max(hist) - min(hist))

    

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_copy[i,j] = hist[img[i,j]]
    
    hist_eq = [0 for i in range(256)]


    # Calculate histogram of equalised image again.
    for i in range(img_copy.shape[0]):
        for j in range(img_copy.shape[1]):
            hist_eq[img_copy[i,j]] += 1

    # plt.subplot(1,2,2)
    # plt.title('After equalizatoin')
    # plt.bar(np.arange(len(hist_eq)),hist_eq)
    # plt.savefig('./results/hist_eq.png')
    # plt.show()
    cv2.imwrite(os.path.join('./results',img_path.split('/')[-1][:-4]+'_histeq_result.png'), img_copy)


    # Comparing results using biultin function

    # built_in_fn_histeq = cv2.equalizeHist(img)
    # # plt.figure()
    # # plt.subplot(1,2,1)
    # # plt.imshow(img_copy, vmin=0, vmax=255)
    # # plt.subplot(1,2,2)
    # # plt.imshow(built_in_fn_histeq, vmin=0, vmax=255)
    # # plt.show()


    # builtin_hist_eq = [0 for i in range(256)]


    # # Calculate histogram of equalised image again.
    # for i in range(img_copy.shape[0]):
    #     for j in range(img_copy.shape[1]):
    #         builtin_hist_eq[built_in_fn_histeq[i,j]] += 1


    # plt.figure()
    # plt.title('Histogram of image')
    # plt.subplot(1,2,1)
    # plt.title('Before equalization')
    # plt.bar(np.arange(len(hist_eq)),hist_eq)
    # plt.subplot(1,2,2)
    # plt.title('After equalizatoin')
    # plt.bar(np.arange(len(builtin_hist_eq)),builtin_hist_eq)
    # # plt.savefig('./results/hist_eq.png')
    # plt.show()

    return img_copy, hist_eq 


def gamma_transform(img_path='./Images/hazy.png', transform_param=1):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

    # plt.figure()
    # plt.title('Before and After gamma')
    # plt.subplot(1,2,1)
    # plt.title('Before Gamma')
    # plt.imshow(img, cmap='gray')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = 255* ((img[i,j]/255)**transform_param)
    
    # plt.subplot(1,2,2)
    # plt.title('After gamma')
    # plt.imshow(img, cmap='gray')
    # plt.show()


    hist = [0 for i in range(256)]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i,j]] += 1


    return img, hist

def question2():

    hazy = cv2.imread('./Images/hazy.png', cv2.IMREAD_GRAYSCALE)
    hist = [0 for i in range(256)]
    for i in range(hazy.shape[0]):
        for j in range(hazy.shape[1]):
            hist[hazy[i,j]] += 1

    hazy_histresult, hazy_hist = histogram_equalization(img_path='./Images/hazy.png')
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.bar(np.arange(len(hist)),hist)
    # plt.subplot(1,2,2)
    # plt.bar(np.arange(len(hazy_hist)),hazy_hist)
    # plt.savefig('./results/hist_hazy.png')
    # plt.show()

    gamma_val = [i*0.1 for i in range(1,51,1)]
    best_gamma_img = hazy
    best_gamma_hist = hist
    best_gamma_val = 1
    best_gamma_mse = 256*256
    MSE = []
    for i,gam in enumerate(gamma_val):
        hazy_gamresult, gamma_hist = gamma_transform(transform_param=gam)
        
        MSE.append(((hazy_histresult-hazy_gamresult)**2).mean())
        if i==0:
            best_gamma_img = hazy_gamresult
            best_gamma_hist = gamma_hist
            best_gamma_val = gam
            best_gamma_mse = MSE[-1]
            continue
        elif MSE[-1] < best_gamma_mse:
            best_gamma_img = hazy_gamresult
            best_gamma_hist = gamma_hist
            best_gamma_val = gam
            best_gamma_mse = MSE[-1]
            
    cv2.imwrite('./results/best_hazy_gamma.png', best_gamma_img)
    cv2.imwrite('./results/histeq_hazy.png', hazy_histresult)

    print('Best Gamma val is: ', best_gamma_val)

    plt.figure()
    plt.title('MSE vs gamma val')
    plt.xlabel('gamma val')
    plt.ylabel('MSE')
    plt.plot(gamma_val,MSE)
    plt.savefig('./results/gamm_vs_mse.png')
    # plt.show()


    plt.figure()
    plt.title('Histogram of image')
    plt.subplot(3,1,1)
    plt.title('Initial Image')
    plt.bar(np.arange(len(hist)),hist)
    plt.subplot(3,1,2)
    plt.title('After histogram equalization')
    plt.bar(np.arange(len(hazy_hist)),hazy_hist)
    plt.subplot(3,1,3)
    plt.title('After best gamma')
    plt.bar(np.arange(len(best_gamma_hist)), best_gamma_hist)
    plt.savefig('./results/bestgamma_hist.png', dpi=600)


    # plt.figure()
    # plt.title('Histogram of image')
    # plt.subplot(2,1,1)
    # plt.title('After histogram equalization')
    # plt.bar(np.arange(len(hazy_hist)),hazy_hist)
    # plt.subplot(2,1,2)
    # plt.title('After best gamma')
    # plt.bar(np.arange(len(best_gamma_hist)), best_gamma_hist)
    # plt.savefig('./results/bestgamma_histeq_hist.png', dpi=600)
    # plt.show()
    cv2.imwrite('./results/bestgamma_img.png', best_gamma_img)


def get_rotated_coordinate(x,y, degree):
    coord = np.matmul(np.array([[np.cos(np.pi * degree/180), -np.sin(np.pi * degree/180)], 
                                            [np.sin(np.pi * degree/180), np.cos(np.pi * degree/180)]]), 
                                            np.array([x,y]))

    return coord

def rotate_image(img, degree_of_rotation=0,interpolation_method='NN'):
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    corners_of_rotated_image = np.array([get_rotated_coordinate(-img.shape[1]/2, img.shape[0]/2, degree_of_rotation),
                                get_rotated_coordinate(-img.shape[1]/2, -img.shape[0]/2, degree_of_rotation),
                                get_rotated_coordinate(img.shape[1]/2, -img.shape[0]/2, degree_of_rotation),
                                get_rotated_coordinate(img.shape[1]/2, img.shape[0]/2, degree_of_rotation)])
    

    x_min, y_min, x_max, y_max = min(corners_of_rotated_image[:,0]), min(corners_of_rotated_image[:,1]), max(corners_of_rotated_image[:,0]), max(corners_of_rotated_image[:,1])
    rotated_img = np.zeros((round(y_max-y_min), round(x_max-x_min)))


    for i in range(rotated_img.shape[0]):
        for j in range(rotated_img.shape[1]):
            if interpolation_method == 'NN':
                # print('inside NN')
                coord = get_rotated_coordinate(rotated_img.shape[1]/2 - j, rotated_img.shape[0]/2 - i, -degree_of_rotation)
                if round(img.shape[0]/2 - coord[1])>=0 and round(img.shape[0]/2 - coord[1])<img.shape[0] and round(img.shape[1]/2 - coord[0])>=0 and round(img.shape[1]/2 - coord[0])<img.shape[1]:
                    rotated_img[i,j] = img[round(img.shape[0]/2 - coord[1]), round(img.shape[1]/2 - coord[0])]

            else:
                # print('Inside bilinear')

                coord = get_rotated_coordinate(rotated_img.shape[1]/2 - j, rotated_img.shape[0]/2 - i, -degree_of_rotation)
                if img.shape[0]/2 - coord[1]>=0 and img.shape[0]/2 - coord[1]<=img.shape[0]-1 and img.shape[1]/2 - coord[0]>=0 and img.shape[1]/2 - coord[0]<=img.shape[1]-1:
                    
                    a = img.shape[0]/2 - coord[1] - int(img.shape[0]/2 - coord[1])
                    b = img.shape[1]/2 - coord[0] - int(img.shape[1]/2 - coord[0])

                    interpolated_x1 = (1-a)*img[int(img.shape[0]/2 - coord[1])+1,int(img.shape[1]/2 - coord[0])] + a*img[int(img.shape[0]/2 - coord[1]),int(img.shape[1]/2 - coord[0])]
                    interpolated_x2 = (1-a)*img[int(img.shape[0]/2 - coord[1])+1,int(img.shape[1]/2 - coord[0])+1] + a*img[int(img.shape[0]/2 - coord[1]),int(img.shape[1]/2 - coord[0])+1]

                    val = (1-b)*interpolated_x2 + b*interpolated_x1


                    rotated_img[i,j] = val#img[round(img.shape[0]/2 - coord[1]), round(img.shape[1]/2 - coord[0])]

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(img, cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(rotated_img, cmap='gray')
    # if interpolation_method=='NN':
    #     cv2.imwrite('./results/NN_single_result.png', rotated_img)
    #     plt.savefig('./results/NN_result.png')
    # else:
    #     cv2.imwrite('./results/bilinear_single_result.png', rotated_img)
    #     plt.savefig('./results/bilinear_result.png')

    # plt.show()

    return rotated_img


def convolution(img, filter):
    
    # Using zero padding

    padded_img = np.zeros((img.shape[0]+ filter.shape[0]-1, img.shape[1]+filter.shape[0]-1))
    padded_img[int(filter.shape[0]/2):-int(filter.shape[0]/2) ,int(filter.shape[0]/2):-int(filter.shape[0]/2)] = img.copy()
    result_img = np.zeros((img.shape[0], img.shape[1]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result_img[i,j] = np.sum(np.multiply(padded_img[i:i+filter.shape[0], j:j+filter.shape[1]], filter))

    return result_img


def Question4(img_path='./Images/study.png', filter_size=5):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)
    # print(img.max())
    # img = img/255 #Normalising the image

    filter = np.ones((5, 5))
    filter = ((1/5)**2) * filter
    unsharp = convolution(img, filter)

    unsharp_mask = img + img-unsharp

    unsharp_mask = np.clip(unsharp_mask,0,255)
    # unsharp_mask = 255*unsharp_mask/unsharp_mask.max()

    cv2.imwrite('./results/unsharp_mask_'+str(filter_size)+'.png', unsharp_mask)

    high_boost = img + 2.5 * (img - unsharp)

    high_boost = np.clip(high_boost, 0,255)
    # high_boost = 255*high_boost/high_boost.max()

    cv2.imwrite('./results/highboost_'+str(filter_size)+'.png', high_boost)




    filter_3 = np.ones((3, 3))
    filter_3 = ((1/3)**2) * filter_3
    blurred_inp = convolution(img, filter_3)
    cv2.imwrite('./results/blurred_inp.png', blurred_inp)

    unsharp_blr = convolution(blurred_inp, filter)
    unsharp_mask = blurred_inp + blurred_inp-unsharp_blr

    unsharp_mask = np.clip(unsharp_mask,0,255)
    # unsharp_mask = 255*unsharp_mask/unsharp_mask.max()

    cv2.imwrite('./results/unsharp_mask_blur'+str(filter_size)+'.png', unsharp_mask)

    high_boost_blr = blurred_inp + 2.5 * (blurred_inp - unsharp_blr)

    high_boost_blr = np.clip(high_boost_blr, 0,255)
    # high_boost = 255*high_boost/high_boost.max()

    cv2.imwrite('./results/highboost_blur_'+str(filter_size)+'.png', high_boost_blr)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(unsharp_mask, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(high_boost_blr, cmap='gray')
    plt.show()

 #####################   Problem 1  ###################
# Question1()

#####################   End of problem 1  ###################   

# histogram_equalization()
#####################   Problem 2  ###################
# question2()

#####################   End of problem 2   ###################


#####################   Problem 3   ###################



# Implementation for rotation by 5 degree clockwise

rotate = rotate_image(cv2.imread('./Images/box.png', cv2.IMREAD_GRAYSCALE),degree_of_rotation=5, interpolation_method='NN')
# rotate = rotate_image(rotate,degree_of_rotation=-30, interpolation_method='NN')

cv2.imwrite('./results/NN_result_5degree.png', rotate)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(rotate, cmap='gray')
rotate = rotate_image(cv2.imread('./Images/box.png', cv2.IMREAD_GRAYSCALE),degree_of_rotation=5, interpolation_method='bilinear')
# rotate = rotate_image(rotate,degree_of_rotation=-30, interpolation_method='bilinear')

cv2.imwrite('./results/bilinear_result_5degree.png', rotate)



plt.subplot(1,2,2)
plt.imshow(rotate, cmap='gray')
plt.show()


# ##################################
# # Find difference between NN image and bilinear image 
# nn_image = cv2.imread('./results/NN_result_5degree.png', cv2.IMREAD_GRAYSCALE)
# bilinear_image = cv2.imread('./results/bilinear_result_5degree.png', cv2.IMREAD_GRAYSCALE)

# diff = nn_image - bilinear_image
# # diff -= diff.min()
# # diff /= diff.max()
# # diff *= 255
# cv2.imwrite('./results/difference_interpolation_5degree.png', diff)





# # Implementation for rotation by 30 degree counter clockwise

# rotate = rotate_image(cv2.imread('./Images/box.png', cv2.IMREAD_GRAYSCALE),degree_of_rotation=-30, interpolation_method='NN')

# cv2.imwrite('./results/NN_result_30degree.png', rotate)

# rotate = rotate_image(cv2.imread('./Images/box.png', cv2.IMREAD_GRAYSCALE),degree_of_rotation=-30, interpolation_method='bilinear')

# cv2.imwrite('./results/bilinear_result_30degree.png', rotate)

# ##################################
# # Find difference between NN image and bilinear image 
# nn_image = cv2.imread('./results/NN_result_30degree.png', cv2.IMREAD_GRAYSCALE)
# bilinear_image = cv2.imread('./results/bilinear_result_30degree.png', cv2.IMREAD_GRAYSCALE)

# diff = nn_image - bilinear_image
# # diff -= diff.min()
# # diff /= diff.max()
# # diff *= 255

# cv2.imwrite('./results/difference_interpolation_30degree.png', diff)



#################    end of problem 3 ########################



#################### Problem 4    #####################
# Question4(filter_size=5)
# Question4(filter_size=3)

#################    end of problem 4 ########################