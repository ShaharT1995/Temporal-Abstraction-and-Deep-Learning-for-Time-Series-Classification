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
    run_list = ["sbatch", sbatch_path] + arguments
    subprocess.Popen(run_list, stdout=temp_file, stderr=temp_file)


# We run this function one time. The function create all the possible combination
def create_combination_lst():
    dict_name = {"archive": ['MTS', 'UCR'],
                 "classifier": ['fcn', 'mlp', 'resnet', 'tlenet', 'twiesn', 'encoder', 'mcdcnn', 'cnn', 'inception',
                                'lstm_fcn', 'mlstm_fcn', 'rocket'],
                 "afterTA": ['True', 'False'],
                 "method": ['RawData', 'sax', 'td4c-cosine', 'gradient'],
                 "combination": ['False'],
                 "transformation": ["1", "2", "3"]}

    keys_list = list(itertools.product(*dict_name.values()))

    combination_lst = []
    for combination in keys_list:
        # If the method is raw data, afterTa must be false, so we remove those combination
        if not (combination[3] == "RawData" and combination[2] == "True" and (combination[5] == "2" or
                                                                              combination[5] == "3")):
            combination_lst.append(list(combination))

    # Save the pickle file
    save_combination_pickle(combination_lst)


# We run this function one time. The function create all the possible combination
def create_combination_lst_2():
    dict_name = {"archive": ['MTS', 'UCR'],
                 "classifier": ['fcn', 'mlp', 'resnet', 'tlenet', 'twiesn', 'encoder', 'mcdcnn', 'cnn', 'inception',
                                'lstm_fcn', 'mlstm_fcn', 'rocket'],
                 "afterTA": ['True', 'False'],
                 "method": ['sax', 'td4c-cosine', 'gradient'],
                 "combination": ['False'],
                 "transformation": ["1", "2", "3"]}

    keys_list = list(itertools.product(*dict_name.values()))

    combination_lst = []
    for combination in keys_list:
        combination_lst.append(list(combination))

    # Save the pickle file
    save_combination_pickle(combination_lst)


def save_combination_pickle(data):
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
    current_user = "shaharap"
    user1 = "hadas5"
    user2 = "roze"
    user3 = "oshermac"

    number_of_total_jobs = 15
    project_path = "/sise/robertmo-group/TA-DL-TSC/"
    sbatch_path = "/sise/home/" + current_user + "/run_python_code"
    current_file_path = "/sise/home/" + current_user + "/run_multi_tasker"
    temp_file = open("/sise/home/" + current_user + "/tmp.txt", 'w')

    # write_pickle("running_dictUCR", {})
    # write_pickle("running_dictMTS", {})
    # create_combination_lst_2()

    while not check_lock():
        print("The file is lock by another user")

    file = open(project_path + "/Run//combination_list.pkl", "rb")
    data = pickle.load(file)

    number_to_run = max(number_of_total_jobs - get_number_of_jobs(current_user), 0)

    combination_for_running = data[: number_to_run]
    save_combination_pickle(data[number_to_run:])

    os.remove(project_path + "/Run//" + current_user + ".txt")
    print("The file unlock by " + current_user)

    for combination in combination_for_running:
        print("Start running: " + str(["run_all"] + combination))
        run_job_using_sbatch(sbatch_path, ["run_all"] + combination)
        time.sleep(60)

    if number_to_run != 0:
        print("Starting the script again")
        run_job_using_sbatch(current_file_path, [])
    else:
        raise Exception("The script finish running")
    exit(0)
