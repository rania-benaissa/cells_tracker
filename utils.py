
from skimage.io import imread
import os
import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt
from IPython.display import HTML


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

    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()

    fig = plt.figure()

    def animate(t):

        plt.cla()
        plt.axis("off")
        plt.imshow(images[t], cmap="gray", animated=True)

    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=10)

    return anim


def animateSequenceVideo(images, interval=500, repeat_delay=1000):
    fig = plt.figure()
    frames = []

    plt.axis("off")

    for image in images:
        frames.append([plt.imshow(image.reshape(
            (image.shape[0], image.shape[1])), cmap="gray", animated=True)])

    anim = matplotlib.animation.ArtistAnimation(fig, frames, interval=interval, blit=True,
                                                repeat_delay=repeat_delay)

    plt.close()
    # Show the animation
    return HTML(anim.to_html5_video())
