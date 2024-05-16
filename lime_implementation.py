import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries
import skimage as skimage

from sklearn.linear_model import Ridge

from lime.lime_tabular import LimeTabularExplainer
from lime.lime_image import LimeImageExplainer

from skimage.segmentation import slic

from sklearn.cluster import KMeans, AgglomerativeClustering

from utils import prediction

def create_explainer_wine(dataset):
    '''
    ADD
    '''

    unbatched_data = dataset.unbatch()
    
    data = np.array([sample[0].numpy() for sample in unbatched_data])
    labels = np.array([sample[1].numpy() for sample in unbatched_data])
    
    return LimeTabularExplainer(
        data,
        mode='classification',
        training_labels=labels,
        feature_names=['fixed acidity', 'volatile acidity', 'citric acid',
                       'residual sugar', 'chlorides', 'free sulfur dioxide',
                       'total sulfur dioxide', 'density', 'pH', 'sulphates',
                       'alcohol'],
        class_names=[f'Quality {i+1}' for i in range(10)]
    ), data, labels

def create_explainer_cifar(dataset):
    '''
    ADD
    '''
    
    unbatched_data = dataset.unbatch()
    
    data = np.array([sample[0].numpy() for sample in unbatched_data])
    labels = np.array([sample[1].numpy() for sample in unbatched_data])
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return LimeImageExplainer(random_state=42), data, labels

def create_explain_pipeline(explainer, model, weights):
    '''
    ADD
    '''
    
    if type(explainer) == LimeImageExplainer:
        return lambda sample : explainer.explain_instance(sample, classifier_fn=lambda data: prediction(weights, model, data), num_features=15, random_seed=42, segmentation_fn=lambda sample: skimage.segmentation.slic(sample, n_segments=15))
        
    elif type(explainer) == LimeTabularExplainer:
        return lambda sample : explainer.explain_instance(data_row=sample, predict_fn=lambda data: prediction(weights, model, data))

    else:
        raise TypeError('Explainer needs to be tabular or image explainer.')

def visualize_lime_img(img, neg_mask, pos_mask, top_label, output_path='./test.jpg'):
    '''
    ADD
    '''
    
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))

    axes[0].imshow(mark_boundaries(img, neg_mask, color=(100, 0, 0), mode='inner'))
    axes[0].set_title(f"Negative Mask for top label: {['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'][top_label]}")
    axes[0].axis("off")

    axes[1].imshow(img)
    axes[1].set_title("Original")
    axes[1].axis("off")

    axes[2].imshow(mark_boundaries(img, pos_mask, color=(100, 100, 0), mode='inner'))
    axes[2].set_title(f"Positive Mask for top label: {['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'][top_label]}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()