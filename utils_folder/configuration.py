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
    perEntity = False

    transformation_number = ""

    def __init__(self):
        self.normalization = False
        self.path = "/sise/robertmo-group/TA-DL-TSC/"
        self.mts_path = self.path + "mtsdata/archives/mts_archive/"
        self.ucr_path = self.path + "UCRArchive_2018/archives/UCRArchive_2018/"
        self.path_after_TA = self.path + "Data/AfterTA/"
        self.path_files_for_TA = self.path + "Data/FilesForTA/"

        self.path_transformation1 = ""
        self.path_transformation2 = ""

        # self.nb_bin = [20]
        self.nb_bin = [3, 5, 10, 20]
        self.std_coefficient = [-1]
        self.max_gap = [1, 2, 3]
        # self.max_gap = [1]
        self.paa_window_size = [1, 2, 5]
        # self.paa_window_size = [1]
        self.gradient_window_size = [10]

        # self.UNIVARIATE_DATASET_NAMES_2018 = ['Beef', 'ACSF1', 'Adiac', 'Computers', 'CricketX', 'CricketY',
        #                                       'CricketZ', 'Crop', 'Earthquakes', 'ECG200', 'ElectricDevices',
        #                                       'EthanolLevel', 'FordA', 'FordB', 'HandOutlines', 'Herring',
        #                                       'LargeKitchenAppliances', 'MiddlePhalanxOutlineCorrect',
        #                                       'MiddlePhalanxTW', 'PhalangesOutlinesCorrect', 'PLAID', 'PowerCons',
        #                                       'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShapesAll',
        #                                       'SmoothSubspace', 'Strawberry', 'SyntheticControl', 'Worms',
        #                                       'WormsTwoClass']

        # Datasets down because EFD
        # 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'GestureMidAirD1', 'GestureMidAirD2',
        # 'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'PLAID', 'ShakeGestureWiimoteZ'

        self.UNIVARIATE_DATASET_NAMES_2018_FailedEFD = ['AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
                                              'GestureMidAirD1', 'GestureMidAirD2',
                                              'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'PLAID',
                                              'ShakeGestureWiimoteZ']

        self.UNIVARIATE_DATASET_NAMES_2018 = ['ACSF1', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME',
                                              'Car', 'CBF', 'Chinatown', 'ChlorineConcentration', 'CinCECGTorso',
                                              'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'Crop',
                                              'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup',
                                              'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'DodgerLoopDay',
                                              'DodgerLoopGame', 'DodgerLoopWeekend', 'Earthquakes',
                                              'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices',
                                              'EOGHorizontalSignal', 'EOGVerticalSignal', 'EthanolLevel', 'FaceAll',
                                              'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB',
                                              'FreezerRegularTrain', 'FreezerSmallTrain', 'Fungi',  'GunPoint',
                                              'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
                                              'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
                                              'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
                                              'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
                                              'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
                                              'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
                                              'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
                                              'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
                                              'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
                                              'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
                                              'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
                                              'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
                                              'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2',
                                              'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShapeletSim', 'ShapesAll',
                                              'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
                                              'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf',
                                              'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2',
                                              'Trace', 'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll',
                                              'UWaveGestureLibraryX', 'UWaveGestureLibraryY',
                                              'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']

        # 'UWaveGestureLibraryZ' - We deleted this data set from the list because the hugobot put all the time series in
        # one bin

        self.UNIVARIATE_DATASET_NAMES_2018 = self.UNIVARIATE_DATASET_NAMES_2018 + self.UNIVARIATE_DATASET_NAMES_2018_FailedEFD

        # self.UNIVARIATE_DATASET_NAMES_2018 = ['UWaveGestureLibraryZ']

        self.MTS_DATASET_NAMES = ['Libras', 'ArabicDigits', 'AUSLAN', 'CharacterTrajectories', 'CMUsubject16', 'ECG',
                                  'JapaneseVowels', 'NetFlow', 'UWave', 'Wafer']
        # self.MTS_DATASET_NAMES = ['Libras', 'ECG']

    def set_path_transformations(self):
        norm = "With ZNorm//" if self.normalization else "Without ZNorm//"

        if self.perEntity:
            self.path_transformation1 = self.path_files_for_TA + "Transformation1 PerEntity//" + norm + \
                                        self.archive + "//"
        else:
            self.path_transformation1 = self.path_files_for_TA + "Transformation1//" + norm + self.archive + "//"

    def set_path_transformations_2(self, nb_bin, paa, max_gap):
        norm = "With ZNorm//" if self.normalization else "Without ZNorm//"

        params = "number_bin-" + str(nb_bin) + "_paa-" + str(paa) + "_max_gap-" + str(max_gap)

        if self.perEntity:
            self.path_transformation2 = self.path_after_TA + "PerEntity" + "//" + norm + self.archive + "//" + \
                                        self.classifier + "//" + self.method + "//" + params + "//"
        else:
            self.path_transformation2 = self.path_after_TA + norm + self.archive + "//" + self.classifier + "//" + \
                                        self.method + "//" + params + "//"

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
    def set_perEntity(perEntity):
        ConfigClass.perEntity = eval(perEntity)

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
