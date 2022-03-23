import itertools
import os
import pickle
import random
import subprocess
import time
import timeit
from datetime import datetime

import pause as pause


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
    dict_name = {"classifier": ['fcn', 'mlp', 'resnet', 'tlenet', 'twiesn', 'encoder', 'mcdcnn', 'cnn', 'inception',
                                'lstm_fcn', 'mlstm_fcn', 'rocket'],
                 "archive": ['MTS', 'UCR'],
                 "afterTA": ['True', 'False'],
                 "method": ['RawData', 'sax', 'td4c-cosine', 'gradient'],
                 "combination": ['False'],
                 "transformation": ["1", "2", "3"]}

    keys_list = list(itertools.product(*dict_name.values()))

    combination_lst = []
    for combination in keys_list:
        # If the method is raw data, afterTa must be false, so we remove those combination
        if not (combination[3] == "RawData" and combination[2] == "True"):
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
    estimated_end_time = timeit.default_timer() + 7 * 24 * 60 * 60
    time_before_sleeping = estimated_end_time - timeit.default_timer()

    current_user = "shaharap"
    user1 = "hadas5"
    user2 = "roze"
    user3 = "oshermac"

    number_of_total_jobs = 2
    project_path = "/sise/robertmo-group/TA-DL-TSC/"
    sbatch_path = "/sise/home/" + current_user + "/run_python_code"
    current_file_path = "/sise/home/" + current_user + "/run_multi_tasker"
    temp_file = open("/sise/home/" + current_user + "/tmp.txt", 'w')

    while time_before_sleeping > 60 * 60:
        while not check_lock():
            print("The file is lock by another user")

        file = open(project_path + "/Run//combination_list.pkl", "rb")
        data = pickle.load(file)

        number_to_run = number_of_total_jobs - get_number_of_jobs(current_user)

        combination_for_running = data[: number_to_run]
        save_combination_pickle(data[number_to_run:])

        os.remove(project_path + "/Run//" + current_user + ".txt")
        print("The file unlock by " + current_user)

        for combination in combination_for_running:
            print("Start running: " + str(["run_all"] + combination))
            run_job_using_sbatch(sbatch_path, ["run_all"] + combination)
            time.sleep(60)

        print("Job went to sleep for 24 hours")
        time.sleep(60 * 60 * 24)

    run_job_using_sbatch(current_file_path, [])
    exit(0)
