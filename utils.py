
from skimage.io import imread, imsave
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def load_folder(images_folder, strat="default"):

    nb_images = len(os.listdir(images_folder))

    images = np.empty(nb_images, dtype="object")

    for i, filename in enumerate(os.listdir(images_folder)):

        stacked_img = imread(os.path.join(images_folder, filename))

        if(strat == "default"):

            images[i] = stacked_img

        if(strat == "all_images"):

            if(i == 0):
                images = stacked_img
            else:
                images = np.vstack((images, stacked_img))

        # print("images nb ", images[i].shape)

    return images


def countImages(images, strat="all_images"):

    if(strat == "all_images"):

        return len(images)

    # preparing for my dataset
    if(strat == "sequences"):
        count = 0

        for i, image in enumerate(images):

            count += len(image)

        return count


def animateSequence(images, interval=500, repeat_delay=1000):
    fig = plt.figure()
    frames = []

    for image in images:
        frames.append([plt.imshow(image, cmap="gray", animated=True)])

    anim = animation.ArtistAnimation(fig, frames, interval=interval, blit=True,
                                     repeat_delay=repeat_delay)
    plt.show()
