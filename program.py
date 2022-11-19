from utils import *

images = load_folder("../Images/01")

print("Nb images ", countImages(images))
animateSequence(images)

images = load_folder(
    "../Images_target/TumorCells")

print("Nb images ", countImages(images))
animateSequence(images[0])
