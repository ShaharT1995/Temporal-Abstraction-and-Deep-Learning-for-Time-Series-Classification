import itertools
import json
import os
import pickle
import random
import subprocess
import time


def get_number_of_jobs(user_name):
    result = subprocess.run(['squeue', '--me'], stdout=subprocess.PIPE)
    number_of_jobs = str(result.stdout).count(user_name) - 1
    return number_of_jobs


# Run batch file with args
def run_job_using_sbatch(sbatch_path, arguments):
    run_list = ["sbatch", "multi_tasker_gpu"] + arguments
    subprocess.Popen(run_list, stdout=temp_file, stderr=temp_file)


def create_combination_per_entity():
    ucr_dict = {"archive": ['UCR'],
                "dataset_name": ['ACSF1', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF',
                                 'Chinatown', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers',
                                 'CricketX', 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction',
                                 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                                 'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'Earthquakes', 'ECG200',
                                 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal',
                                 'EOGVerticalSignal', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
                                 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain', 'Fungi',
                                 'GunPoint', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
                                 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
                                 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
                                 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat',
                                 'Meat', 'MedicalImages', 'MelbournePedestrian', 'MiddlePhalanxOutlineAgeGroup',
                                 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MixedShapesRegularTrain',
                                 'MixedShapesSmallTrain', 'MoteStrain', 'NonInvasiveFetalECGThorax1',
                                 'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect',
                                 'Phoneme', 'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
                                 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
                                 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'Rock',
                                 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2',
                                 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace',
                                 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry',
                                 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2',
                                 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll',
                                 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer',
                                 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'],
                 "nb_bins": ['3', '5', '10', '20'],
                 "method": ['sax', 'td4c-cosine', 'gradient', 'equal-frequency', 'equal-width'],
                 "combination": ['False']}

    mts_dict = {"archive": ['MTS'],
                "dataset_name": ['Libras', 'ArabicDigits', 'CharacterTrajectories', 'ECG',
                                  'JapaneseVowels', 'NetFlow', 'Wafer'],
                 "nb_bins": ['3', '5', '10', '20'],
                 "method": ['sax', 'td4c-cosine', 'gradient', 'equal-frequency', 'equal-width'],
                 "combination": ['False']}

    keys_list = list(itertools.product(*ucr_dict.values())) + list(itertools.product(*mts_dict.values()))

    combination_lst = []
    for combination in keys_list:
        # Gradient method cannot be with gradient combination
        if not (combination[3] == "gradient" and combination[4] == "True"):
            combination_lst.append(list(combination))

    # Save the pickle file
    save_combination_pickle(combination_lst)
    print()


# for step one running hugobot
def create_combination_list():
    # dict_name = {"archive": ['UCR', 'MTS'],
    #              "classifier": ['fcn', 'mlp', 'resnet', 'twiesn', 'encoder', 'mcdcnn', 'cnn', 'inception',
    #                             'lstm_fcn', 'mlstm_fcn', 'rocket'],
    #              "afterTA": ['True', 'False'],
    #              "method": ['RawData', 'sax', 'td4c-cosine', 'gradient'],
    #              "combination": ['False', 'True'],
    #              "transformation": ['1'],
    #              "perEntity": ['True', 'False']}

    dict_name = {"archive": ['MTS'],
                 "classifier": ['HugoBotFiles'],
                 "afterTA": ['True'],
                 "method": ['sax', 'gradient', 'equal-frequency', 'equal-width'],
                 "combination": ['False'],
                 "transformation": ['1'],
                 "perEntity": ['False']}

    keys_list = list(itertools.product(*dict_name.values()))

    combination_lst = []
    for combination in keys_list:
        # combination[3] - Method, combination[2] - afterTA, combination[4] - Gradient combination,
        # combination[5] - Transformation number, combination[6] - Per entity

        # Raw data cannot be with afterTA
        if not (combination[3] == "RawData" and combination[2] == "True"):
            # Raw data cannot be with per entity
            if not (combination[3] == "RawData" and combination[6] == "True"):
                # Raw data cannot be with gradient combination
                if not (combination[3] == "RawData" and combination[4] == "True"):
                    # TA method cannot be without afterTA
                    if not (combination[3] != "RawData" and combination[2] == "False"):
                        # Raw data can be only with transformation 1 (without gradient combination, per entity, and TA)
                        if not (combination[3] == "RawData" and combination[2] == "False" and combination[6] == "False"
                                and combination[4] == "False" and (combination[5] == "2" or combination[5] == "3")):
                            # Gradient method cannot be with gradient combination
                            if not (combination[3] == "gradient" and combination[4] == "True"):
                                combination_lst.append(list(combination))

    # Save the pickle file
    save_combination_pickle(combination_lst)
    print()


# We run this function one time. The function create all the possible combination
def create_combination_gpu():
    dict_name = {"archive": ['UCR', 'MTS'],
                 "classifier": ['fcn', 'resnet', 'inception', 'mcdcnn', 'mlstm_fcn', 'cnn', 'mlp'],
                 "afterTA": ['False', 'True'],
                 "method": ['sax', 'gradient', 'equal-frequency', 'equal-width', 'RawData'],
                 "combination": ['False'],
                 "transformation": ['1', '2'],
                 "perEntity": ['False']}

    keys_list = list(itertools.product(*dict_name.values()))

    combination_lst = []
    for combination in keys_list:
        # combination[3] - Method, combination[2] - afterTA, combination[4] - Gradient combination,
        # combination[5] - Transformation number, combination[6] - Per entity

        # Raw data cannot be with afterTA
        if not (combination[3] == "RawData" and combination[2] == "True"):
            # Raw data cannot be with per entity
            if not (combination[3] == "RawData" and combination[6] == "True"):
                # Raw data cannot be with gradient combination
                if not (combination[3] == "RawData" and combination[4] == "True"):
                    # TA method cannot be without afterTA
                    if not (combination[3] != "RawData" and combination[2] == "False"):
                        # Raw data can be only with transformation 1 (without gradient combination, per entity, and TA)
                        if not (combination[3] == "RawData" and combination[2] == "False" and combination[6] == "False"
                                and combination[4] == "False" and (combination[5] == "2" or combination[5] == "3")):
                            # Gradient method cannot be with gradient combination
                            if not (combination[3] == "gradient" and combination[4] == "True"):
                                combination_lst.append(list(combination))
    # Save the pickle file
    save_combination_pickle(combination_lst, gpu=True)


# We run this function one time. The function create all the possible combination
def create_combination_cpu():
    dict_name = {"archive": ['UCR'],
                 "classifier": ['twiesn', 'rocket'],
                 "afterTA": ['True'],
                 "method": ['sax'],
                 "combination": ['False'],
                 "transformation": ['1', '2', '3'],
                 "perEntity": ['True']}

    # dict_name = {"archive": ['UCR', 'MTS'],
    #              "classifier": ['twiesn', 'rocket'],
    #              "afterTA": ['False', 'True'],
    #              "method": ['sax', 'td4c-cosine', 'gradient', 'RawData'],
    #              "combination": ['False', 'True'],
    #              "transformation": ['1', '2', '3'],
    #              "perEntity": ['False']}

    keys_list = list(itertools.product(*dict_name.values()))

    combination_lst = []
    for combination in keys_list:
        # combination[3] - Method, combination[2] - afterTA, combination[4] - Gradient combination,
        # combination[5] - Transformation number, combination[6] - Per entity

        # Raw data cannot be with afterTA
        if not (combination[3] == "RawData" and combination[2] == "True"):
            # Raw data cannot be with per entity
            if not (combination[3] == "RawData" and combination[6] == "True"):
                # Raw data cannot be with gradient combination
                if not (combination[3] == "RawData" and combination[4] == "True"):
                    # TA method cannot be without afterTA
                    if not (combination[3] != "RawData" and combination[2] == "False"):
                        # Raw data can be only with transformation 1 (without gradient combination, per entity, and TA)
                        if not (combination[3] == "RawData" and combination[2] == "False" and combination[6] == "False"
                                and combination[4] == "False" and (combination[5] == "2" or combination[5] == "3")):
                            # Gradient method cannot be with gradient combination
                            if not (combination[3] == "gradient" and combination[4] == "True"):
                                combination_lst.append(list(combination))

    # Save the pickle file
    save_combination_pickle(combination_lst, gpu=False)


def save_combination_pickle(data, gpu=None):
    if gpu is not None:
        type = "gpu" if gpu else "cpu"
        file = open(project_path + "/Run//combination_list_" + type + ".pkl", "wb")
    else:
        file = open(project_path + "/Run//combination_list.pkl", "wb")
    pickle.dump(data, file)
    file.close()


def check_lock():
    # Check if another user don't reading the file, sleep until the file is unlock
    while os.path.exists(project_path + "/Run//" + user1 + ".txt") or \
            os.path.exists(project_path + "/Run//" + user2 + ".txt") or \
            os.path.exists(project_path + "/Run//" + user3 + ".txt"):
        time.sleep(random.randint(1, 10))

    # The file is unlock, so we wrote a file with the user that read the file
    with open(project_path + "/Run//" + current_user + ".txt", 'w') as f:
        f.write("")

    # Wait 5 seconds and check if another user open the file. We do this to be sure that two users don't open the
    # file exactly at the same time (the same second)
    time.sleep(5)
    if os.path.exists(project_path + "/Run//" + user1 + ".txt") or \
            os.path.exists(project_path + "/Run//" + user2 + ".txt") or \
            os.path.exists(project_path + "/Run//" + user3 + ".txt"):
        # If two users does open the file at the same time, they give up on the file, and try again
        os.remove(project_path + "/Run//" + current_user + ".txt")
        time.sleep(random.randint(1, 10))

        return False

    print("File is lock by " + current_user)
    return True


def write_pickle(name, data):
    file = open("/sise/robertmo-group/TA-DL-TSC/Project/temporal_abstraction_f/pickle_files/" + name + ".pkl", "wb")
    pickle.dump(data, file)
    file.close()


if __name__ == '__main__':
    current_user = "shaharap"
    user1 = "hadas5"
    user2 = "roze"
    user3 = "oshermac"

    number_of_total_jobs = 15
    project_path = "/sise/robertmo-group/TA-DL-TSC/"
    sbatch_path = "/sise/home/" + current_user + "/run_python_code_gpu"
    current_file_path = "/sise/home/" + current_user + "/run_multi_tasker_gpu"
    temp_file = open("/sise/home/" + current_user + "/tmp.txt", 'w')

    # create_combination_gpu()
    # create_combination_cpu()
    create_combination_list()
    # create_combination_per_entity()

    # # For step one - CPU
    # write_pickle("create_files_dict_UCR", {})
    # write_pickle("create_files_dict_MTS", {})

    # For step two - GPU
    # write_pickle("running_dictUCR", {})
    # write_pickle("running_dictMTS", {})
    # write_pickle("run_to_do_UCR", {})
    # write_pickle("run_to_do_MTS", {})

    # while not check_lock():
    #     print("The file is lock by another user")
    #
    # file = open(project_path + "/Run//combination_list.pkl", "rb")
    # data = pickle.load(file)
    # print(len(data))

    # file = open(project_path + "/Run//combination_list_gpu.pkl", "rb")
    # data1 = pickle.load(file)
    # print(len(data1))
    #
    # file = open(project_path + "Project/temporal_abstraction_f/pickle_files//create_files_dict_MTS.pkl", "rb")
    # data2 = pickle.load(file)
    # print(len(data2))
    #
    # file = open(project_path + "Project/temporal_abstraction_f/pickle_files//hugobot_dict.pkl", "rb")
    # data3 = pickle.load(file)
    # print(len(data3))

    # file = open(project_path + "Project/temporal_abstraction_f/pickle_files//running_dictUCR.pkl", "rb")
    # data2 = pickle.load(file)
    # print(len(data2))
    #
    # file = open(project_path + "Project/temporal_abstraction_f/pickle_files//running_dictMTS.pkl", "rb")
    # data3 = pickle.load(file)

    # del data3[('MTS', 'mcdcnn', 'td4c-cosine', 10, 1, -1, 1, None, '3', True, False)]
    # write_pickle("running_dictMTS", data3)

    # file = open(project_path + "Project/temporal_abstraction_f/pickle_files//run_to_do_MTS.pkl", "rb")
    # data2 = pickle.load(file)
    # print(len(data2))
    #
    # file = open(project_path + "Project/temporal_abstraction_f/pickle_files//run_to_do_UCR.pkl", "rb")
    # data3 = pickle.load(file)
    # print(len(data3))

    # print()
    #
    # number_to_run = max(number_of_total_jobs - get_number_of_jobs(current_user), 0)
    #
    # combination_for_running = data[: number_to_run]
    # save_combination_pickle(data[number_to_run:], gpu=True)
    #
    # os.remove(project_path + "/Run//" + current_user + ".txt")
    # print("The file unlock by " + current_user)
    #
    # for combination in combination_for_running:
    #     print("Start running: " + str(["run_all"] + combination))
    #     run_job_using_sbatch(sbatch_path, ["run_all"] + combination)
    #     time.sleep(60)
    #
    # if len(data[number_to_run:]) != 0:
    #     print("Starting the script again")
    #     run_job_using_sbatch(current_file_path, [])
    #     time.sleep(60 * 60 * 2)
    # else:
    #     raise Exception("The script finish running")
    # exit(0)
