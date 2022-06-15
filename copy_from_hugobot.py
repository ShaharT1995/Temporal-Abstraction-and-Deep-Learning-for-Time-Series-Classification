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


source_folder = r"/sise/robertmo-group/TA-DL-TSC/Data/AfterTA/MTS/HugoBotFiles/"
for classifier in ['fcn', 'mlp', 'resnet', 'twiesn', 'encoder', 'mcdcnn', 'cnn', 'inception', 'lstm_fcn', 'mlstm_fcn',
                   'rocket']:
    destination_folder = r"/sise/robertmo-group/TA-DL-TSC/Data/AfterTA/MTS/" + classifier + "/"
    copytree(source_folder, destination_folder)
    print(classifier + " Done")

