import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering

from lime import lime_image
from PIL import Image
import skimage as skimage

import os
import argparse

import numpy as np

def skimage_segmentation(image : np.array, n_segments : int) -> tuple:
    img = Image.fromarray(np.array(image))
    mask = skimage.segmentation.slic(np.array(img), n_segments=n_segments)
    color_map = {i+1 : np.random.randint(0,255,3).tolist() for i in range(n_segments)}
    rgb_array = np.array([color_map[i] for i in mask.reshape(-1)]).reshape((*mask.shape, 3)).astype('uint8')
    return image, mask, rgb_array

def display_images_side_by_side(img1, img2, title):
    fig, ax = plt.subplots(1, 2)
    
    # Display img1
    ax[0].imshow(img1)
    # ax[0].axis('off')  # No axes for img1
    
    # Display img2
    ax[1].imshow(img2)
    # ax[1].axis('off')  # No axes for img2
    
    # Set the title for the figure
    plt.suptitle(title)
    
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def segment_only(img, mask, seg=1, replace=[0,0,0]):
    segment = np.zeros(img.shape).astype('uint8')
    for i in np.ndindex(img.shape[:2]):
        if mask[i] == seg:
            segment[i] = img[i]

    return segment

def plot_segments(original, mask, segments):
    # Calculate the number of additional images
    n = len(segments) + 2

    # Create a grid for plotting
    ncols = 4
    nrows = n//4 if n % 4 == 0 else n//4+1

    # Create a figure and axes
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 12))

    # Plot additional images
    c = -2
    for x in range(nrows):
        for y in range(ncols):
            axs[x,y].axis('off')
            
            if c == -2:
                axs[x,y].imshow(original)
                
            elif c == -1:
                axs[x,y].imshow(mask)
                
            elif c <= len(segments):
                axs[x,y].imshow(segments[c])
                
            else:
                axs[x,y].imshow(np.full((original.shape), 255).astype('uint8'))
            c += 1
            
    plt.savefig('./img/PoggoRoggo1.jpg')
    plt.show()