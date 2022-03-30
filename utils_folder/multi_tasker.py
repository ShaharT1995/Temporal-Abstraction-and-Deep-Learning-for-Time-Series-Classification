import itertools
import os
import pickle
import random
import subprocess
import time
from utils_folder.utils import write_pickle
import timeit


def get_number_of_jobs(user_name):
    result = subprocess.run(['squeue', '--me'], stdout=subprocess.PIPE)
    number_of_jobs = str(result.stdout).count(user_name) - 1
    return number_of_jobs


# Run batch file with args
def run_job_using_sbatch(sbatch_path, arguments):
    run_list = ["sbatch", "multi_tasker_cpu"] + arguments
    subprocess.Popen(run_list, stdout=temp_file, stderr=temp_file)

# for step one running hugobot
def create_combination_list():
    dict_name = {"archive": ['UCR', 'MTS'],
                 "classifier": ['fcn', 'mlp', 'resnet', 'tlenet', 'twiesn', 'encoder', 'mcdcnn', 'cnn', 'inception',
                                'lstm_fcn', 'mlstm_fcn', 'rocket'],
                 "afterTA": ['True'],
                 "method": ['sax', 'td4c-cosine', 'gradient'],
                 "combination": ['False'],
                 "transformation": ["1", "2", "3"]}

    keys_list = list(itertools.product(*dict_name.values()))

    combination_lst = []
    for combination in keys_list:
        combination_lst.append(list(combination))

    # Save the pickle file
    save_combination_pickle(combination_lst)


# We run this function one time. The function create all the possible combination
def create_combination_gpu():
    dict_name = {"archive": ['UCR', 'MTS'],
                 "classifier": ['fcn', 'mlp', 'resnet', 'tlenet', 'encoder', 'mcdcnn', 'cnn', 'inception', 'lstm_fcn',
                                'mlstm_fcn'],
                 "afterTA": ['False', "True"],
                 "method": ['sax', 'td4c-cosine', 'gradient', "RawData"],
                 "combination": ['False'],
                 "transformation": ["1", "2", "3"]}

    keys_list = list(itertools.product(*dict_name.values()))

    combination_lst = []
    for combination in keys_list:
        if not (combination[3] == "RawData" and combination[2] == "True" and (combination[5] == "2" or
                                                                              combination[5] == "3")):
            combination_lst.append(list(combination))
    # Save the pickle file
    save_combination_pickle(combination_lst, gpu=True)


# We run this function one time. The function create all the possible combination
def create_combination_cpu():
    dict_name = {"archive": ['UCR', 'MTS'],
                 "classifier": ['rocket', 'twiesn'],
                 "afterTA": ['False', "True"],
                 "method": ['sax', 'td4c-cosine', 'gradient', "RawData"],
                 "combination": ['False'],
                 "transformation": ["1", "2", "3"]}

    keys_list = list(itertools.product(*dict_name.values()))

    combination_lst = []
    for combination in keys_list:
        if not (combination[3] == "RawData" and combination[2] == "True" and (combination[5] == "2" or
                                                                              combination[5] == "3")):
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


if __name__ == '__main__':
    current_user = "hadas5"
    user1 = "shaharap"
    user2 = "roze"
    user3 = "oshermac"

    number_of_total_jobs = 15
    project_path = "/sise/robertmo-group/TA-DL-TSC/"
    sbatch_path = "/home/" + current_user + "/run_python_code_cpu"
    current_file_path = "/home/" + current_user + "/run_multi_tasker_cpu"
    temp_file = open("/home/" + current_user + "/tmp.txt", 'w')

    write_pickle("running_dictUCR", {})
    write_pickle("running_dictMTS", {})
    create_combination_list()

    # while not check_lock():
    #     print("The file is lock by another user")
    #
    # file = open(project_path + "/Run//combination_list.pkl", "rb")
    # data = pickle.load(file)
    #
    # number_to_run = max(number_of_total_jobs - get_number_of_jobs(current_user), 0)
    #
    # combination_for_running = data[: number_to_run]
    # save_combination_pickle(data[number_to_run:])
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
    #     time.sleep(30)
    # else:
    #     raise Exception("The script finish running")
    # exit(0)
