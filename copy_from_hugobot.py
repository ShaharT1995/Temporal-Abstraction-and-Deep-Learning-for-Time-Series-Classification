import os
import shutil


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            return None
        return directory_path


for bins in [3, 5, 10, 20]:
    for max_gap in [1, 2, 3]:
        for method in ["equal-frequency", "equal-width", "sax"]:
            params = "number_bin-" + str(bins) + "_paa-1_max_gap-" + str(max_gap)
            source_folder = r"/sise/robertmo-group/TA-DL-TSC/Data/AfterTA/PerEntity/Without ZNorm/UCR/HugoBotFiles/" \
                            + method + "/" + params + "/"
            for classifier in ['fcn', 'resnet', 'inception', 'mcdcnn', 'mlstm_fcn', 'cnn', 'mlp']:
                print("Start: " + classifier)
                destination_folder = r"/sise/robertmo-group/TA-DL-TSC/Data/AfterTA/PerEntity/Without ZNorm/UCR/" \
                            + classifier + "/" + method + "/" + params + "/"
                create_directory(destination_folder)

                copytree(source_folder, destination_folder)
                print("Done: " + classifier)


# source_folder = r"/sise/robertmo-group/TA-DL-TSC/Data/AfterTA/PerEntity/Without ZNorm/UCR/HugoBotFiles/"
# for classifier in ['fcn', 'resnet', 'inception', 'mcdcnn', 'mlstm_fcn', 'cnn', 'mlp']:
#     print("Start: " + classifier)
#     destination_folder = r"/sise/robertmo-group/TA-DL-TSC/Data/AfterTA/PerEntity/Without ZNorm/UCR/" + classifier + "/"
#     create_directory(destination_folder)
#     copytree(source_folder, destination_folder)
#     print("Done: " + classifier)
