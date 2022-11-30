from utils import load_folder
from skimage.measure import regionprops


class Dataset(object):

    """Dataset class"""

    def __init__(self):

        super(Dataset, self).__init__()

        self.train_images = []
        self.train_masks = []
        self.train_bbs = []
        self.train_cmasks = []
        self.train_ids = []

        self.test_images = []
        self.test_masks = []
        self.test_bbs = []
        self.test_cmasks = []
        self.test_ids = []

        # to show bbox
        #bx = (minc, maxc, maxc, minc, minc)
        # by = (minr, minr, maxr, maxr, minr) 
        # plt.plot(bx, by, '-b', linewidth=0.5)

    def loadTrain(self, imgs_path="", masks_path="", track_path=""):

        self.train_images, self.train_masks, self.train_bbs, self.train_cmasks, self.train_ids = self.load_dataset(
            imgs_path, masks_path)

        return self.train_images, self.train_masks

    def loadTest(self, imgs_path="", masks_path="", track_path=""):

        self.test_images, self.test_masks, self.test_bbs, self.test_cmasks, self.test_ids = self.load_dataset(
            imgs_path, masks_path)

        return self.test_images, self.test_masks

    def load_dataset(self, imgs_path, masks_path):

        if(imgs_path != ""):

            images = load_folder(imgs_path)

        if(masks_path != ""):

            masks = load_folder(masks_path)
            bbs, cmasks, ids = self.extractInfos(masks)

        # if(track_path != ""):

        #     tracks = load_folder(track_path)

        return images, masks, bbs, cmasks, ids

    def extractInfos(self, masks):

        bbs = []

        cropped_masks = []

        ids = []

        for mask in masks:
            # for all the image
            bb = []
            id_imgs = []
            cropped_mask = []

            regions = regionprops(mask)

            for props in regions:
                # this part is for bb
                minr, minc, maxr, maxc = props.bbox

                minr -= 2
                minc -= 2
                maxr -= 2
                maxc -= 2

                bb.append((minr, minc, maxr, maxc))

                # this part is for ID + cropped masks

                id_imgs.append(props.label)
                cropped_mask.append(props.image_filled)

            bbs.append(bb)
            ids.append(id_imgs)
            cropped_masks.append(cropped_mask)

        return bbs, cropped_masks, ids
