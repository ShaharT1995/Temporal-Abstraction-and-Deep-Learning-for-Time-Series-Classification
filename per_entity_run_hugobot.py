def run_per_entity(nb_bin, dataset_name):
    prop_path = config.path_files_for_TA + "PerEntity//" + config.archive + "//" + config.classifier + "//" +\
                config.method + "//"

    create_directory(prop_path)

    # Make the 3 files - gkb.csv, ta.csv and ppa.csv
    print("Step 2: make the gkb.csv, ta.csv and ppa.csv \n")

    if not check_pickle_exists("PerEntity_Dict_" + config.archive):
        write_pickle("PerEntity_Dict_" + config.archive, {})

    if not check_pickle_exists("run_to_do_per_entity_" + config.archive):
        write_pickle("run_to_do_per_entity_" + config.archive, {})

    running_dict = open_pickle("PerEntity_Dict_" + config.archive)

    config.set_path_transformations_2(nb_bin)

    for std in config.std_coefficient:
        for max_gap in config.max_gap:
            if config.method == "gradient":
                for gradient_window in config.gradient_window_size:
                    running_dict = run_hugobot(config, dataset_name, prop_path, running_dict, max_gap, config.method, nb_bin,
                                               config.paa_window_size, std, gradient_window)
            else:
                running_dict = run_hugobot(config, dataset_name, prop_path, running_dict, max_gap, config.method, nb_bin,
                                           config.paa_window_size, std)
    print("Done")


def run_hugobot(config, dataset_name, path, running_dict, max_gap, method, nb_bin, paa, std, gradient_window=None):
    print("--------------------------------------------------------------------------------------------------------")
    print(dataset_name + " - Method: " + method + ", Bins: " + str(nb_bin) + " Combination: " + str(config.combination))
    print("--------------------------------------------------------------------------------------------------------\n")

    key = (config.archive, dataset_name, config.classifier, method, nb_bin, paa, std, max_gap, gradient_window,
           config.combination, config.perEntity)

    # if key in running_dict:
    #     print("Already Done! \n")
    #     return running_dict

    if False:
        print()

    else:
        prop_path = path + "number_bin_" + str(nb_bin) + "//"
        create_directory(prop_path)

        create_three_files(config=config,
                           path=prop_path,
                           method=method,
                           nb_bins=nb_bin,
                           paa_window_size=paa,
                           std_coefficient=std,
                           max_gap=max_gap,
                           gradient_window_size=gradient_window)

        # print("Hugobot is OFF")
        print("Step 3: run hugobot")
        run_cli(config, prop_path, max_gap)

        if config.combination and config.method != "gradient":
            print("Step 3.1: make the gkb.csv, ta.csv and ppa.csv for " + method + " method\n")

            gradient_prop_path = config.path_files_for_TA
            if config.perEntity:
                gradient_prop_path += "PerEntity//"
            gradient_prop_path += config.archive + "//" + config.classifier + "//gradient//number_bin_"\
                                  + str(nb_bin) + "//"

            create_directory(gradient_prop_path)

            # TODO Change the gradient window size
            create_three_files(config=config,
                               path=gradient_prop_path,
                               method="gradient",
                               nb_bins=nb_bin,
                               paa_window_size=paa,
                               std_coefficient=std,
                               max_gap=max_gap,
                               gradient_window_size=config.gradient_window_size[0])

            method = config.method
            config.set_method("gradient")

            running_dict = open_pickle("PerEntity_Dict_" + config.archive)
            gradient_key = (config.archive, dataset_name, config.classifier, method, nb_bin, paa, std, max_gap,
                            gradient_window, config.combination, config.perEntity)
            if gradient_key not in running_dict:
                print("Gradient not done! need to wait until this files will over")
                not_done = open_pickle("run_to_do_per_entity_" + config.archive)
                not_done[gradient_key] = True
                write_pickle("run_to_do_per_entity_" + config.archive, not_done)

                exit(1)

            # print("Step 3.2: run hugobot for Gradient method")
            # run_cli(config, gradient_prop_path, max_gap)

            config.set_method(method)

            print("Step 4: transformation 2")
            combining_two_methods_ucr(config, prop_path, gradient_prop_path) if config.archive == "UCR" else \
                combining_two_methods_mts(config, prop_path, gradient_prop_path)

        else:
            # Make the second temporal abstraction -> hugobot output files to original format
            print("Step 4: transformation 2")
            new_ucr_files(config, prop_path) if config.archive == "UCR" else new_mts_files(config, prop_path)

        running_dict = open_pickle("PerEntity_Dict_" + config.archive)
        running_dict[key] = True
        write_pickle("PerEntity_Dict_" + config.archive, running_dict)
        return running_dict


if __name__ == '__main__':
    import sys

    sys.path.insert(0, '/sise/robertmo-group/TA-DL-TSC/Project/')

    from utils_folder.configuration import ConfigClass
    from utils_folder.utils import write_pickle, open_pickle, create_directory, check_pickle_exists

    config = ConfigClass()
    config.set_archive(sys.argv[2])

    from temporal_abstraction_f.set_parameters import create_three_files
    from temporal_abstraction_f.tensor_transformation import new_ucr_files, new_mts_files
    from temporal_abstraction_f.combination_gradient_state import combining_two_methods_ucr, combining_two_methods_mts

    sys.path.insert(0, '/sise/robertmo-group/TA-DL-TSC/Project/Hugobot')
    from Hugobot.cli import run_cli

    if sys.argv[1] == 'create_files':
        # create_files UCR SmoothSubspace 3 sax False
        config.set_classifier("HugoBotFiles")
        config.set_afterTA("True")
        config.set_method(sys.argv[5])
        config.set_combination(sys.argv[6])
        config.set_transformation("1")
        config.set_perEntity("True")

        config.set_path_transformations()

        if config.afterTA:
            if config.archive == "UCR":
                config.UNIVARIATE_DATASET_NAMES_2018 = [sys.argv[3]]
            else:
                config.MTS_DATASET_NAMES = [sys.argv[3]]
            nb_bins = int(sys.argv[4])
            run_per_entity(nb_bins, sys.argv[3])

    else:
        print("Something wrong with the parameters!")
