import os
import numpy as np
import random as rn
import tensorflow as tf
from tensorflow.python.keras import backend as K


class ConfigClass:
    method = ""
    classifier = ""
    archive = ""
    afterTA = False
    combination = False

    transformation_number = ""

    def __init__(self):
        self.path = "/sise/robertmo-group/TA-DL-TSC/"
        self.mts_path = self.path + "mtsdata/archives/mts_archive/"
        self.ucr_path = self.path + "UCRArchive_2018/archives/UCRArchive_2018/"
        self.path_after_TA = self.path + "Data/AfterTA/"
        self.path_files_for_TA = self.path + "Data/FilesForTA/"

        self.path_transformation1 = ""
        self.path_transformation2 = ""

        self.nb_bin = [3, 10]
        self.std_coefficient = [-1]
        self.max_gap = [1]
        self.paa_window_size = 1
        self.gradient_window_size = [10]
        self.UNIVARIATE_DATASET_NAMES_2018 = ['Worms']
        #todo -'Coffee'

        # self.UNIVARIATE_DATASET_NAMES_2018 = ['Beef', 'ACSF1', 'Adiac', 'Beef', 'Computers', 'CricketX', 'CricketY', 'CricketZ',
        #                                       'Crop', 'Earthquakes', 'ECG200', 'ElectricDevices', 'EthanolLevel',
        #                                       'FordA', 'FordB', 'HandOutlines', 'Herring', 'LargeKitchenAppliances',
        #                                       'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
        #                                       'PhalangesOutlinesCorrect', 'PLAID', 'PowerCons', 'SemgHandMovementCh2',
        #                                       'SemgHandSubjectCh2', 'ShapesAll', 'SmoothSubspace', 'Strawberry',
        #                                       'SyntheticControl', 'Worms', 'WormsTwoClass']

        self.MTS_DATASET_NAMES = ['Libras', 'ArabicDigits', 'AUSLAN', 'CharacterTrajectories', 'CMUsubject16', 'ECG',
                                  'JapaneseVowels', 'KickvsPunch', 'NetFlow', 'UWave', 'Wafer', 'WalkvsRun']

        #self.MTS_DATASET_NAMES = ['ArabicDigits', 'AUSLAN']

    def set_path_transformations(self):
        self.path_transformation1 = self.path_files_for_TA + "Transformation1//" + self.archive + "//"
        self.path_transformation2 = self.path_after_TA + self.archive + "//" + self.classifier + "//" + self.method \
                                    + "//"

    def set_path_transformations_2(self, nb_bin ):
        self.path_transformation2 = self.path_after_TA + self.archive + "//" + self.classifier + "//" + self.method \
                                    + "//" + "number_bin_" + str(nb_bin) + "//"
    @staticmethod
    def set_method(method):
        ConfigClass.method = method

    @staticmethod
    def set_transformation(transformation_number):
        ConfigClass.transformation_number = transformation_number

    @staticmethod
    def set_classifier(cf):
        ConfigClass.classifier = cf

    @staticmethod
    def set_archive(archive):
        ConfigClass.archive = archive
        ConfigClass.ITERATIONS = 10 if (ConfigClass.archive == "MTS") else 5

    @staticmethod
    def set_afterTA(after_TA):
        ConfigClass.afterTA = eval(after_TA)

    @staticmethod
    def set_combination(combination):
        ConfigClass.combination = eval(combination)

    @staticmethod
    def set_seed():
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        # Setting the seed for numpy-generated random numbers
        np.random.seed(37)

        # Setting the seed for python random numbers
        rn.seed(1254)

        # Setting the graph-level random seed.
        tf.random.set_seed(89)


        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        K.set_session(sess)

