import os
import shutil


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


source_folder = r"/sise/robertmo-group/TA-DL-TSC/Data/AfterTA/UCR - Without Normalization/HugoBotFiles - Without ZNorm/equal-frequency"
for classifier in ['resnet']:
    print("Start: " + classifier)
    destination_folder = r"/sise/robertmo-group/TA-DL-TSC/Data/AfterTA/Without ZNorm//UCR//" + classifier + \
                         "//equal-frequency"
    copytree(source_folder, destination_folder)
    print("Done: " + classifier)

