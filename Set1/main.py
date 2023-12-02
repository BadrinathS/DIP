import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def histogram(img_path='./Images/coins.png'):
    # To compute histograms


    hist = [0 for i in range(256)]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i,j]] += 1
    
    plt.figure()
    plt.hist(img.ravel(), bins=256)
    plt.savefig('./results/problem1_hist.png')
    # plt.show()
    
    hist_avg_intensity = 0
    for i in range(len(hist)):
        hist_avg_intensity += i*hist[i]
    hist_avg_intensity = hist_avg_intensity/(img.shape[0]*img.shape[1])

    print("Average Intensity calculated from histogram: ", hist_avg_intensity)
    print("\n Average Intensity calculated from Image: ", np.mean(img.ravel()))

    return hist



def between_class_variance(img_path = './Images/coins.png'):
    

    hist = [0 for i in range(256)]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i,j]] += 1
    
    prob_hist = [i/(img.shape[0]*img.shape[1]) for i in hist]

    max_intra_class_variance = 0        #Maximum variance initialization

    for threshold in range(256):

        prob_class1 = sum(prob_hist[0:threshold+1])
        prob_class2 = sum(prob_hist[threshold+1:])

        if prob_class1==0 or prob_class2==0:
            continue

        mean_class1 = 0
        mean_class2 = 0

        for i in range(256):
            if i>threshold:
                mean_class2 += i*prob_hist[i]
            
            else:
                mean_class1 += i*prob_hist[i]

        mean_class1 = mean_class1/prob_class1
        mean_class2 = mean_class2/prob_class2

        intra_class_variance = prob_class1*prob_class2*((mean_class1-mean_class2)**2)

        if intra_class_variance > max_intra_class_variance:
            max_intra_class_variance = intra_class_variance
            max_intra_class_variance_thres = threshold

    # cv2.imwrite('./results/within_class_binary_image.png', np.where(img>min_intra_class_variance_thres,255,0))
    return max_intra_class_variance_thres


def within_class_variance(img_path = './Images/coins.png'):

    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    hist = [0 for i in range(256)]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i,j]] += 1
    
    prob_hist = [i/(img.shape[0]*img.shape[1]) for i in hist]# hist/(img.shape[0]*img.shape[1])

    min_inter_class_variance = 256*256 #Maximum possible variance

    for threshold in range(256):

        prob_class1 = sum(prob_hist[0:threshold+1])
        prob_class2 = sum(prob_hist[threshold+1:])

        if prob_class1==0 or prob_class2==0:
            continue

        mean_class1 = 0
        mean_class2 = 0
        var_class1 = 0
        var_class2 = 0

        for i in range(256):
            if i>threshold:
                mean_class2 += i*prob_hist[i]
            
            else:
                mean_class1 += i*prob_hist[i]

        mean_class1 = mean_class1/prob_class1
        mean_class2 = mean_class2/prob_class2

        for i in range(256):
            if i > threshold:
                var_class2 += ((i-mean_class2)**2)*prob_hist[i]
            
            else:
                var_class1 += ((i-mean_class1)**2)*prob_hist[i]
        
        var_class1 = var_class1/prob_class1
        var_class2 = var_class2/prob_class2

        inter_class_variance = prob_class1*var_class1 + prob_class2*var_class2

        # print(inter_class_variance, intra_class_variance)
        if inter_class_variance<=min_inter_class_variance:
            min_inter_class_variance = inter_class_variance
            min_inter_class_variance_thres = threshold
        
    # cv2.imwrite('./results/between_class_binary_image.png', np.where(img>max_inter_class_variance_thres,255,0))
    return min_inter_class_variance_thres


def image_superimpose(txt_img_path='./Images/IIScText.png', depth_img_path='./Images/IIScTextDepth.png', bkg_img_path='./Images/IIScMainBuilding.png'):
    txt_img = cv2.imread(txt_img_path)
    depth_img = cv2.imread(depth_img_path, cv2.IMREAD_GRAYSCALE)
    bkg_img = cv2.imread(bkg_img_path)

    print("Shape of txt image: ", txt_img.shape)
    print("Shape of depth image: ", depth_img.shape)
    print("Shape of bkg image: ", bkg_img.shape)

    binary_threshold = within_class_variance(depth_img_path)
    print("Threshold computed for superimpose is: ", binary_threshold)
    # print(binary_threshold)
    for i in range(txt_img.shape[0]):
        for j in range(txt_img.shape[1]):
            if depth_img[i,j] >= binary_threshold:
                bkg_img[i,j,0] = txt_img[i,j,0]
                bkg_img[i,j,1] = txt_img[i,j,1]
                bkg_img[i,j,2] = txt_img[i,j,2]
    cv2.imwrite('./results/IIScText_binarised_problem3.png', np.where(depth_img>=binary_threshold, 255,0))
    cv2.imwrite('./results/result_problem3.png', bkg_img)


def count_characters(img_path='./Images/quote.png',threshold=-1):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if threshold <0:
        threshold = within_class_variance(img_path)
    print('Threshold is: ', threshold)
    binary_image = img.copy()
    binary_image = np.where(img>threshold, 0, 255)
    print(binary_image[10,12])

    
    region_count = 1
    copy_img = np.zeros(binary_image.shape)

    equivalency_list = {}

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if binary_image[i,j]==255:
                if i==0 and j==0:
                    # if j==0:
                    copy_img[i,j] = region_count
                    region_count += 1
                elif i==0:
                    if copy_img[i,j-1] != 0:
                        copy_img[i,j] = copy_img[i,j-1]
                    else:
                        copy_img[i,j] = region_count
                        region_count += 1
                
                elif j==0:
                    if copy_img[i-1,j] != 0:
                        copy_img[i,j] = copy_img[i-1,j]
                    else:
                        copy_img[i,j] = region_count
                        region_count += 1
                
                else:
                    if copy_img[i-1,j] == 0 and copy_img[i,j-1]==0:
                        copy_img[i,j] = region_count
                        region_count += 1
                    elif copy_img[i-1,j]==0 and copy_img[i,j-1]!=0:
                        copy_img[i,j] = copy_img[i,j-1]
                    elif copy_img[i-1,j]!=0 and copy_img[i,j-1]==0:
                        copy_img[i,j] = copy_img[i-1,j]
                    elif copy_img[i-1,j]!=0 and copy_img[i,j-1]!=0 and copy_img[i-1,j]==copy_img[i,j-1]:
                        copy_img[i,j] = copy_img[i-1,j]
                    else:
                        # if max(copy_img[i-1,j], copy_img[i,j-1]) not in equivalency_list.keys():
                        copy_img[i,j] = min(copy_img[i-1,j], copy_img[i,j-1])
                        equivalency_list[max(copy_img[i-1,j], copy_img[i,j-1])] = min(copy_img[i-1,j], copy_img[i,j-1])
                        # else:
                        #     copy_img[i,j] = equivalency_list[max(copy_img[i-1,j], copy_img[i,j-1])]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if copy_img[i,j] in equivalency_list.keys():
                copy_img[i,j] = equivalency_list[copy_img[i,j]]
    
    #Count connected components
    connected_components = set()
    size_of_connected_component = {}
    for i in range(copy_img.shape[0]):
        for j in range(copy_img.shape[1]):
            if copy_img[i,j] == 0:
                continue
            if copy_img[i,j] in size_of_connected_component.keys():
                size_of_connected_component[copy_img[i,j]] += 1
            else:
                size_of_connected_component[copy_img[i,j]] = 1
            if copy_img[i,j] in connected_components:
                continue
            else:
                connected_components.add(copy_img[i,j])

    total_connected_components = 0
    keys = []
    for key, val in size_of_connected_component.items():
        if val>=10:
            total_connected_components+=1
        else:
            keys.append(key)
    for key in keys:
        del size_of_connected_component[key]
    
    cv2.imwrite('./results/characters.png', copy_img)
    cv2.imwrite('./results/binary.png', binary_image)
    cv2.imwrite('./results/counting.png', np.where(copy_img>0,255,0))
    print("Character count is: ",total_connected_components)
    return len(connected_components), size_of_connected_component




# def MSER(imp_path='./Images/Characters.png'):


#     for threshold in range(256):
#         total_characters, length_of_each_character = count_characters(img_path, threshold)







#####################       ######################
#problem 1
# histogram()
# 
#####################       ######################




#####################       ######################
#Problem 2
# start = time.time()
# intra = within_class_variance()
# end = time.time()
# print("Within Class variance threshold: ", intra, " Time taken: ", end-start)

# start = time.time()
# inter = between_class_variance()
# end = time.time()
# print("Between Class variance threshold: ", inter, " Time taken: ", end-start)
# exit()
#####################       ######################


#####################       ######################
#problem 3
# image_superimpose()
# exit()
#####################       ######################


#####################       ######################
#problem 4
count_characters()
exit()
#####################       ######################

#####################       ######################
#problem 5
# MSER()
#####################       ######################