import cv2
import numpy as np
import psychopy.filters
import rasterio
from scipy.ndimage import label
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu



def airport_fls_extraction(input_image_filepath, output_mask_filepath, RGB = True, blur_kernel = 5, butworth_cutoff = 0.04, 
                           order = 1, return_all_coords = True):
    """
    This function is used to implement the First Layer Saliency (FLS) model for the extraction of airport candidate regions
    from a given satellite image as proposed in the paper 'Airport Detection and Aircraft Recognition Based on Two - Layer
    Saliency Model in High Spatial Resolution Remote - Sensing Images' by Zhang L., Zhang Y. (2017)
    
    Inputs:
    - input_image_filepath: File path of image where the airport regions are to be extracted
    - output_mask_filepath: File path of output binary mask indicating where the airport region is
    - RGB: Boolean indicating whether first 3 bands of image is RGB or BGR
    - blur_kernel: Kernel size for the application of the gaussian blur to the two - chromatic channels (RG and BY)
    - butworth_cutoff: Cutoff frequency for n - order Butterworth high - pass filter
    - order: Order of the Butterworth high - pass filter
    - return_all_coords: Boolean indicating whether to return the geo - referenced coordinates of the detected airport region
                         with respect to the coordinate reference system of the input image raster
    
    Outputs:
    - airport_map: Binary mask indicating the location of detected airport region
    - (cmin, rmin, cmax, rmax): Raster coordinates of the bounding box of the detected airport region in the image, in the 
                                following order: (column minimum, row minimum, column maximum, row maximum)
    - (coord_long_min, coord_lat_max, coord_long_max, coord_lat_min): Map coordinates of the bounding box of the detected 
                                                                      airport region in the image (with respect to the 
                                                                      coordinate reference system of the image), in the 
                                                                      following order: (minimum longitude, maximum latitude, 
                                                                      maximum longitude, minimum latitude)
    
    """
    
    with rasterio.open(input_image_filepath) as f:
        metadata = f.profile
        img_orig = np.uint8(rescale_intensity(np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0]), 
                                              out_range = 'uint8'))
    img = cv2.pyrUp(cv2.pyrDown(img_orig))
        
        
    if RGB:
        lab = cv2.cvtColor(img[:, :, 0 : 3], cv2.COLOR_RGB2LAB)
    else:
        lab = cv2.cvtColor(img[:, :, 0 : 3], cv2.COLOR_BGR2LAB)
        
    for band in range(1, 3):
        lab[:, :, band] = cv2.GaussianBlur(lab[:, :, band], (blur_kernel, blur_kernel), 0)
        
    
    
    l_freq = np.fft.fft2(lab[:, :, 0])

    hp_filt = psychopy.filters.butter2d_hp(size = lab[:, :, 0].shape, cutoff = butworth_cutoff, n = order)
    l_filt = np.fft.fftshift(l_freq) * hp_filt
    l_new = np.uint8(rescale_intensity(np.real(np.fft.ifft2(np.fft.ifftshift(l_filt))), out_range = 'uint8'))
    
    lab[:, :, 0] = l_new
    
    
    
    L = np.abs(np.mean(lab[:, :, 0]) - lab[:, :, 0])
    A = np.abs(np.mean(lab[:, :, 1]) - lab[:, :, 1])
    B = np.abs(np.mean(lab[:, :, 2]) - lab[:, :, 2])
    
    saliency_map = L + A + B
    
    
    
    thresholded = saliency_map > threshold_otsu(saliency_map)
    connected_region, num_objects = label(thresholded)
    
    
    
    size = np.zeros((len(np.unique(connected_region)) - 1), dtype = np.uint16)
    
    for region in range(1, len(np.unique(connected_region))):
        size[region - 1] = np.sum(connected_region == region)
    
    candidate_region = size.argsort()[-5 :] + 1
    
    
    
    saliency_candidate = np.zeros((5), dtype = np.uint16)
    for i in range(5):
        saliency_candidate[i] = np.sum(saliency_map * (connected_region == candidate_region[i]))

    airport_map = (connected_region == candidate_region[np.argmax(saliency_candidate)]).astype(metadata['dtype'])
    
    
    
    rows = np.any(airport_map, axis = 1)
    cols = np.any(airport_map, axis = 0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    
    
    ulx = metadata['transform'][2]
    xres = metadata['transform'][0]
    uly = metadata['transform'][5]
    yres = metadata['transform'][4]
    lrx = ulx + (img.shape[1] * xres)                                                         
    lry = uly - (img.shape[0] * abs(yres))
    xf = ((img.shape[1]) ** 2 / (img.shape[1] + 1)) / (lrx - ulx)
    yf = ((img.shape[0]) ** 2 / (img.shape[0] + 1)) / (lry - uly)

    coord_long_max = (cmax / xf) + ulx
    coord_long_min = (cmin / xf) + ulx
    coord_lat_max = (rmin / yf) + uly
    coord_lat_min = (rmax / yf) + uly
    
    
    metadata['count'] = 1
    with rasterio.open(output_mask_filepath, 'w', **metadata) as dst:
        dst.write(np.transpose(np.expand_dims(airport_map, axis = 2), [2, 0, 1]))
        
    
    if return_all_coords == False:
        return airport_map, (cmin, rmin, cmax, rmax)
    else:
        return airport_map, (cmin, rmin, cmax, rmax), (coord_long_min, coord_lat_max, coord_long_max, coord_lat_min)
    