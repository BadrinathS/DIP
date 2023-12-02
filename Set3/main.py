import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import cv2


def Problem1(f0=50):

    M = 500
    

    img = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            img[i,j] = 255*np.cos(2*np.pi*f0*np.sqrt((i-float(M/2))**2 + (j - float(M/2))**2)/M)

    # plt.figure()
    # plt.imshow(img, cmap='gray')
    # plt.show()


    fft = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft)

    # fshift = fft

    # magnitude = 20*np.log(np.abs(fshift))
    magnitude = np.abs(fshift)
    print(magnitude.max(), magnitude.min())

    magnitude = 255*(magnitude - magnitude.min())/(magnitude.max() - magnitude.min())

    # f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(fft))

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(magnitude, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(img_back, cmap='gray')
    plt.show()



    cv2.imwrite('./results/original_' + str(f0) + '.png', img)
    cv2.imwrite('./results/fft2_magnitude_' + str(f0) + '.png', magnitude)
    cv2.imwrite('./results/ifft2_magnitude_' + str(f0) + '.png', img_back)



def Problem2(img_path = './Files/characters.tif', d0=100, filter='ideal'):
    img = np.float32(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))

    img_shift = img.copy()

    for i in range(img_shift.shape[0]):
        for j in range(img_shift.shape[1]):
            img_shift[i,j] =  img_shift[i,j] * (-1)**(i+j)

    # print(img_shift.max(), img_shift.min(), img_shift.dtype)
    fft = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft)

    fft_img = np.abs(fshift)


    fft_imgshift = np.fft.fft2(img_shift)
    # fshift_imgshift = np.fft.fftshift(fft_imgshift)

    fft_img_imgshift = np.abs(fft_imgshift)

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(img, cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(fft_img_imgshift, cmap='gray')
    # plt.show()
    
    # print(fft_img.max(), fft_img.min(), fft_img.mean())

    cv2.imwrite('./results/fft_characters_' + str(d0) + '.png', fft_img)

    if filter == 'ideal':
        for i in range(fft_img.shape[0]):
            for j in range(fft_img.shape[1]):
                if np.sqrt((i-float(img.shape[0])/2)**2 + (j - float(img.shape[1])/2)**2) > d0:
                    fft_img[i,j] = 0
                    fshift[i,j] = complex(0,0)

                    fft_imgshift[i,j] = complex(0,0)
                    fft_img_imgshift[i,j] = 0
                    # fft = 0 + 0

        
        img_back = np.abs(np.fft.ifft2(fshift))
        img_back_shift = np.abs(np.fft.ifft2(fft_imgshift))

        cv2.imwrite('./results/fft_afteridealLPF_characters_' + str(d0) + '.png', fft_img_imgshift)

        for i in range(img_back_shift.shape[0]):
            for j in range(img_back_shift.shape[1]):
                img_back[i,j] =  img_back[i,j] * (-1)**(i+j)
                img_back_shift[i,j] =  img_back_shift[i,j] * (-1)**(i+j)
        

        cv2.imwrite('./results/reconstructed_afteridealLPF_characters_' + str(d0) + '.png', img_back)

    if filter == 'gaussian':
        for i in range(fft_img.shape[0]):
            for j in range(fft_img.shape[1]):
                # if np.sqrt((i-float(img.shape[0])/2)**2 + (j - float(img.shape[1])/2)**2) > d0:
                fft_img[i,j] = fft_img[i,j]*np.exp(-((i-float(img.shape[0])/2)**2 + (j - float(img.shape[1])/2)**2)/(2*d0*d0))
                fshift[i,j] = fshift[i,j]*np.exp(-((i-float(img.shape[0])/2)**2 + (j - float(img.shape[1])/2)**2)/(2*d0*d0))

                fft_imgshift[i,j] = fft_imgshift[i,j]*np.exp(-((i-float(img.shape[0])/2)**2 + (j - float(img.shape[1])/2)**2)/(2*d0*d0))
                fft_img_imgshift[i,j] = fft_img_imgshift[i,j]*np.exp(-((i-float(img.shape[0])/2)**2 + (j - float(img.shape[1])/2)**2)/(2*d0*d0))
                    # fft = 0 + 0

        
        img_back = np.abs(np.fft.ifft2(fshift))
        img_back_shift = np.abs(np.fft.ifft2(fft_imgshift))

        cv2.imwrite('./results/fft_aftergaussianLPF_characters_' + str(d0) + '.png', fft_img_imgshift)

        for i in range(img_back_shift.shape[0]):
            for j in range(img_back_shift.shape[1]):
                img_back[i,j] =  img_back[i,j] * (-1)**(i+j)
                img_back_shift[i,j] =  img_back_shift[i,j] * (-1)**(i+j)
        

        cv2.imwrite('./results/reconstructed_aftergaussianLPF_characters_' + str(d0) + '.png', img_back)


def Problem3_inverse(threshold=0.1):
    img_lownoise = cv2.imread('./Files/Blurred_LowNoise.png', cv2.IMREAD_GRAYSCALE)
    img2_highnoise = cv2.imread('./Files/Blurred_HighNoise.png', cv2.IMREAD_GRAYSCALE)

    
    kernel = scipy.io.loadmat('./Files/BlurKernel.mat')['h']
    
    kernel_padded = np.zeros(img_lownoise.shape)
    kernel_padded[:kernel.shape[0], :kernel.shape[1]] = kernel.copy()

    kernel_shifted = np.zeros(img_lownoise.shape)

    # print(np.unravel_index(np.argmax(kernel), kernel.shape))

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            # print(i,j,(i-kernel.shape[0]//2)%img_lownoise.shape[0], (j-kernel.shape[1]//2)%img_lownoise.shape[1])
            kernel_shifted[(i-9)%img_lownoise.shape[0], (j-24)%img_lownoise.shape[1]] = kernel_padded[i,j]
        
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(kernel_padded)
    # plt.subplot(1,2,2)
    # plt.imshow(kernel_shifted)
    # plt.show()
    # exit()
    
    cv2.imwrite('./results/original_kernel.png', kernel_padded)
    cv2.imwrite('./results/shifted_kernel.png', kernel_shifted)

    kernel_fft = np.fft.fft2(kernel_shifted)
    fft_kernel_img = np.abs(kernel_fft)

    inverse_fft_img = fft_kernel_img.copy()
    inverse_fft = kernel_fft.copy()

    for i in range(fft_kernel_img.shape[0]):
        for j in range(fft_kernel_img.shape[1]):
            if fft_kernel_img[i,j] < threshold:
                inverse_fft_img[i,j] = 0
                inverse_fft[i,j] = complex(0,0)
            else:
                inverse_fft[i,j] = 1/kernel_fft[i,j]
                inverse_fft_img[i,j] = 1/fft_kernel_img[i,j]

    cv2.imwrite('./results/kernel_fft.png', 255*((fft_kernel_img- fft_kernel_img.min())/(fft_kernel_img.max() - fft_kernel_img.min())))
    cv2.imwrite('./results/inverted_kernel_fft.png', 255*((inverse_fft_img- inverse_fft_img.min())/(inverse_fft_img.max() - inverse_fft_img.min())))
    print(fft_kernel_img.max(), fft_kernel_img.min())
    print(inverse_fft_img.max(), inverse_fft_img.min())

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(fft_kernel_img, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(inverse_fft_img, cmap='gray')
    plt.show()

    fft_img_lownoise = np.fft.fft2(img_lownoise)
    fft_img_highnoise = np.fft.fft2(img2_highnoise)

    fft_img_lownoise_temp = np.multiply(fft_img_lownoise, inverse_fft)
    fft_img_highnoise_temp = np.multiply(fft_img_highnoise, inverse_fft)

    fft_img_lownoise_recon = np.fft.ifft2(fft_img_lownoise_temp)
    fft_img_highnoise_recon = np.fft.ifft2(fft_img_highnoise_temp)

    fft_img_lownoise_recon_img = np.abs(fft_img_lownoise_recon)
    fft_img_highnoise_recon_img = np.abs(fft_img_highnoise_recon)



    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(img_lownoise, cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(fft_img_lownoise_recon_img, cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(img2_highnoise, cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(fft_img_highnoise_recon_img, cmap='gray')
    plt.show()

    cv2.imwrite('./results/recon_lownoise.png', fft_img_lownoise_recon_img)
    cv2.imwrite('./results/recon_highnoise.png', fft_img_highnoise_recon_img)


def Problem3_weiner():
    img_lownoise = cv2.imread('./Files/Blurred_LowNoise.png', cv2.IMREAD_GRAYSCALE)
    img2_highnoise = cv2.imread('./Files/Blurred_HighNoise.png', cv2.IMREAD_GRAYSCALE)

    
    
    kernel = scipy.io.loadmat('./Files/BlurKernel.mat')['h']
    
    kernel_padded = np.zeros(img_lownoise.shape)
    kernel_padded[:kernel.shape[0], :kernel.shape[1]] = kernel.copy()

    kernel_shifted = np.zeros(img_lownoise.shape)

    # print(np.unravel_index(np.argmax(kernel), kernel.shape))

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            # print(i,j,(i-9)%img_lownoise.shape[0], (j-24)%img_lownoise.shape[1])
            kernel_shifted[(i-9)%img_lownoise.shape[0], (j-24)%img_lownoise.shape[1]] = kernel_padded[i,j]

    # exit()

    kernel_fft = np.fft.fft2(kernel_shifted)
    fft_kernel_img = np.abs(kernel_fft)

    # Low noise inverse degredation

    fft_img_lownoise = np.fft.fft2(img_lownoise)
    fft_img_highnoise = np.fft.fft2(img2_highnoise)

    H_hat_low = np.array(np.zeros(fft_img_lownoise.shape), dtype = 'complex_')
    for i in range(fft_kernel_img.shape[0]):
        for j in range(fft_kernel_img.shape[1]):
            H_hat_low[i,j] = np.conjugate(kernel_fft[i,j])/(fft_kernel_img[i,j]**2 + (np.sqrt((i%fft_kernel_img.shape[0])**2 + (j%fft_kernel_img.shape[1])**2)/(10**5)))
    
    H_hat_high = np.array(np.zeros(fft_img_lownoise.shape), dtype = 'complex_')
    for i in range(fft_kernel_img.shape[0]):
        for j in range(fft_kernel_img.shape[1]):
            H_hat_high[i,j] = np.conjugate(kernel_fft[i,j])/(fft_kernel_img[i,j]**2 + (np.sqrt((i%fft_kernel_img.shape[0])**2 + (j%fft_kernel_img.shape[1])**2)/(10**4)))
    

    fft_img_lownoise_temp = np.multiply(fft_img_lownoise, H_hat_low)

    fft_img_lownoise_recon = np.fft.ifft2(fft_img_lownoise_temp)
    fft_img_lownoise_recon_img = np.abs(fft_img_lownoise_recon)



    fft_img_highnoise_temp = np.multiply(fft_img_highnoise, H_hat_high)

    fft_img_highnoise_recon = np.fft.ifft2(fft_img_highnoise_temp)
    fft_img_highnoise_recon_img = np.abs(fft_img_highnoise_recon)

    cv2.imwrite('./results/weiner_recon_img_lownoise.png', fft_img_lownoise_recon_img)
    cv2.imwrite('./results/weiner_recon_img_highnoise.png', fft_img_highnoise_recon_img)
    

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(img_lownoise, cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(fft_img_lownoise_recon_img, cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(img2_highnoise, cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(fft_img_highnoise_recon_img, cmap='gray')
    plt.show()


    # fft_img_highnoise_recon = np.fft.ifft2(fft_img_highnoise_temp)

    
    # fft_img_highnoise_recon_img = np.abs(fft_img_highnoise_recon)



##############   Problem1 ######### 
# Problem1()
# Problem1(f0=20)
# Problem1(f0=80)

##############  End of Problem1 ######### 

##############   Problem 2 #############
# Problem2(filter='ideal')
# Problem2(filter='gaussian')

# Problem2(d0=50, filter='ideal')
# Problem2(d0=50, filter='gaussian')

##############   End of Problem 2 #############


##############   Problem 3 #############
Problem3_inverse()
Problem3_weiner()