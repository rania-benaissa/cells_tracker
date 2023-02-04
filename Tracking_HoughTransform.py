import numpy as np
import cv2
import os
import datetime
roi_defined = False


def computeGradient(image, threshold=0.5):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    Gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    Gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

    magni = np.sqrt(Gx**2 + Gy**2)

    # converting it to gradient
    ori = np.arctan2(Gy, Gx) * 180 // np.pi

    unmarked_indices = np.where(magni >= threshold)

    ori_bgr = cv2.cvtColor(ori.astype('float32'), cv2.COLOR_GRAY2BGR)

    ori_bgr = cv2.normalize(ori_bgr, None, 0, 1,
                            cv2.NORM_MINMAX)

    # i normalize it here to keep the red color

    ori_bgr[np.where(magni < threshold)] = [0, 0, 1.0]

    return magni, ori, ori_bgr, unmarked_indices


def create_rTable(ori, r, c, h, w, unmarked_indices):

    # is a dictionary of lists (orientation,list of vectors)
    rTable = {}

    roi_centroid = np.array([r + (h // 2), c + (w // 2)])

    # creating the rTable
    for x, y in zip(unmarked_indices[0], unmarked_indices[1]):

        vect_dist = [roi_centroid - np.array([y + r, x + c])]

        # it the key doesn t exist already we create it
        if(ori[x, y] not in rTable):

            rTable[ori[x, y]] = list()

        rTable[ori[x, y]].extend(vect_dist)

    return rTable


def computeHT(ori, rTable, unmarked_indices):

    hough_image = np.zeros(ori.shape)

    for x, y in zip(unmarked_indices[0], unmarked_indices[1]):

        angle = ori[x, y]

        # if the key exists
        if angle in rTable:

            vectors = rTable[angle]

            for vect in vectors:

                if (y + vect[0] >= 0 and y + vect[0] < hough_image.shape[1]) and (x + vect[1] >= 0 and x + vect[1] < hough_image.shape[0]):

                    hough_image[x + vect[1], y + vect[0]] += 1

    return hough_image


def define_ROI(event, x, y, flags, param):
    global r, c, w, h, roi_defined
    # if the left mouse button was clicked,
    # record the starting ROI coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    # if the left mouse button was released,
    # record the ROI coordinates and dimensions
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2 - r)
        w = abs(c2 - c)
        r = min(r, r2)
        c = min(c, c2)
        roi_defined = True


video_name = 'VOT-Ball'
TEST_SAVE_DIR = 'saved_images_hough'

threshold = 20

cap = cv2.VideoCapture(video_name + '.mp4')


# saving folder

if not os.path.exists(TEST_SAVE_DIR):
    os.makedirs(TEST_SAVE_DIR)

submit_dir = video_name + \
    "_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
submit_dir = os.path.join(TEST_SAVE_DIR, submit_dir)
os.makedirs(submit_dir)


# take first frame of the video
ret, frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)


# # keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("First image", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the ROI is defined, draw it!
    if (roi_defined):
        # draw a green rectangle around the region of interest
        cv2.rectangle(frame, (r, c), (r + h, c + w), (0, 255, 0), 2)
    # else reset the image...
    else:
        frame = clone.copy()
    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break


track_window = (r, c, h, w)
# set up the ROI for tracking
roi = frame[c:c + w, r:r + h]

magni, ori, thresholded, unmarked_indices = computeGradient(roi, threshold)

# init the rTable
rTable = create_rTable(ori, r, c, h, w, unmarked_indices)


cv2.imwrite(submit_dir + '/Frame_0.png', frame)


cpt = 1
while(1):
    ret, frame = cap.read()
    if ret == True:

        magni, ori, thresholded, unmarked_indices = computeGradient(
            frame, threshold)

        hough_image = computeHT(ori, rTable, unmarked_indices)

        # argmax computation:
        posy, posx = np.unravel_index(np.argmax(hough_image), ori.shape)

        r, c = max(posx - (h // 2), 0), max(posy - (w // 2), 0)

        frame_tracked = cv2.rectangle(
            frame, (r, c), (r + h, c + w), (0, 255, 0), 2)

        # normalise images before displaying it
        hough_image = cv2.normalize(
            hough_image, None, 0, 1, cv2.NORM_MINMAX)

        ori = cv2.normalize(ori, None, 0, 1,
                            cv2.NORM_MINMAX)

        magni = cv2.normalize(magni, None, 0, 1,
                              cv2.NORM_MINMAX)

        # Plotting all images
        cv2.imshow('Frame', frame_tracked)
        cv2.imshow("Magnitude", magni)
        cv2.imshow("orientation", ori)
        cv2.imshow("ori with threshold = " + str(threshold), thresholded)
        cv2.imshow("Hough transform", hough_image)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        # elif k == ord('s'):

        # normalise to save
        hough_image = cv2.normalize(
            hough_image, None, 0, 255, cv2.NORM_MINMAX)

        ori = cv2.normalize(ori, None, 0, 255,
                            cv2.NORM_MINMAX)

        magni = cv2.normalize(magni, None, 0, 255,
                              cv2.NORM_MINMAX)

        thresholded = cv2.normalize(thresholded, None, 0, 255,
                                    cv2.NORM_MINMAX)

        cv2.imwrite(submit_dir + '/Frame_%04d.png' % cpt, frame_tracked)
        cv2.imwrite(submit_dir + '/magnitude_%04d.png' % cpt, magni)
        cv2.imwrite(submit_dir + '/orientation_%04d.png' % cpt, ori)
        cv2.imwrite(submit_dir + '/thresholded_ori_%04d.png' %
                    cpt, thresholded)
        cv2.imwrite(submit_dir + '/hough_%04d.png' % cpt, hough_image)

        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()
