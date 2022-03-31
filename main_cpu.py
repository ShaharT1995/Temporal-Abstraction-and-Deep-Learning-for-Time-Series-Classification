import os


def create_files():
    # Make the first temporal abstraction -> original data sets to hugobot format
    print("Step 1: transformation 1")

    if config.archive == "UCR":
        uni_ta_1 = UnivariateTA1(config, 0)
        next_attribute = uni_ta_1.convert_all_UTS()
    # config.archive == "MTS"
    else:
        multi_ta_1 = MultivariateTA1(config, 0)
        next_attribute = multi_ta_1.convert_all_MTS()

    write_pickle("next_property_index" + config.archive, {"ID": next_attribute})

    print()


def run_cpu():
    prop_path = config.path_files_for_TA + config.archive + "//" + config.classifier + "//" + config.method + "//"

    if config.combination:
        prop_path += "combination//"
    create_directory(prop_path)

    # Make the 3 files - gkb.csv, ta.csv and ppa.csv
    print("Step 2: make the gkb.csv, ta.csv and ppa.csv \n")

    if not check_pickle_exists("create_files_dict_" + config.archive):
        write_pickle("create_files_dict_" + config.archive, {})

    running_dict = open_pickle("create_files_dict_" + config.archive)

    for nb_bin in config.nb_bin:
        for std in config.std_coefficient:
            for max_gap in config.max_gap:
                if config.method == "gradient":
                    for gradient_window in config.gradient_window_size:
                        running_dict = run_hugobot(config, prop_path, running_dict, max_gap, config.method, nb_bin,
                                                   config.paa_window_size, std, gradient_window)
                else:
                    running_dict = run_hugobot(config, prop_path, running_dict, max_gap, config.method, nb_bin,
                                               config.paa_window_size, std)
    print("Done")


def run_hugobot(config, prop_path, running_dict, max_gap, method, nb_bin, paa, std, gradient_window=None):
    # TODO - COMBINATION
    # todo - Add gradient to the print
    print("-------------------------------------------------------------------------------------")
    print("Method: " + method + ", Bins: " + str(nb_bin) + ", PAA: " + str(paa) + ", STD: " +
          str(std) + ", Max_Gap: " + str(max_gap))
    print("------------------------------------------------------------------------------------- \n")

    key = (config.archive, config.classifier, method, nb_bin, paa, std, max_gap, gradient_window,
           config.transformation_number, config.combination)

    transformation_dict = open_pickle("transformation_number")
    key_2 = (config.archive, config.classifier, method, nb_bin, paa, std, max_gap, gradient_window, config.combination)

    if key in running_dict:
        print("Already Done! \n")
        return running_dict

    else:
        prop_path += "number_bin_" + str(nb_bin) + "//"
        create_directory(prop_path)
        #todo - remove this if
        if key_2 not in transformation_dict:
            create_three_files(config=config,
                               path=prop_path,
                               method=method,
                               nb_bins=nb_bin,
                               paa_window_size=paa,
                               std_coefficient=std,
                               max_gap=max_gap,
                               gradient_window_size=gradient_window)

            print("Step 3: run hugobot")
            run_cli(config, prop_path, max_gap)

            if config.combination and config.method != "gradient":
                print("Step 3.1: make the gkb.csv, ta.csv and ppa.csv for " + method + " method\n")

                gradient_prop_path = config.path_files_for_TA + config.archive + "//" + config.classifier + "//gradient//"
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

                print("Step 3.2: run hugobot for Gradient method")
                run_cli(config, gradient_prop_path, max_gap)

                config.set_method(method)

                combining_two_methods(config, prop_path)

            else:
                transformation_dict[key_2]=True
                write_pickle("transformation_number" , transformation_dict)


        else:
            # Make the second temporal abstraction -> hugobot output files to original format
            print("Step 4: transformation 2")
            config.set_path_transformations_2(nb_bin)
            new_ucr_files(config, prop_path) if config.archive == "UCR" else new_mts_files(config, prop_path)

        running_dict[key] = True
        write_pickle("create_files_dict_" + config.archive, running_dict)
        return running_dict


if __name__ == '__main__':
    import sys

    sys.path.insert(0, '/sise/robertmo-group/TA-DL-TSC/Project/')

    from utils_folder.configuration import ConfigClass
    from utils_folder.utils import write_pickle, open_pickle, create_directory, check_pickle_exists, \
        transform_mts_to_ucr_format

    config = ConfigClass()
    config.set_archive(sys.argv[2])

    from temporal_abstraction_f.multivariate_ta_1 import MultivariateTA1
    from temporal_abstraction_f.univariate_ta_1 import UnivariateTA1
    from temporal_abstraction_f.set_parameters import create_three_files
    from temporal_abstraction_f.multivariate_ta_2 import new_mts_files
    from temporal_abstraction_f.univariate_ta_2 import new_ucr_files
    from temporal_abstraction_f.ucr_to_mts import combining_two_methods

    sys.path.insert(0, '/sise/robertmo-group/TA-DL-TSC/Project/Hugobot')
    from Hugobot.cli import run_cli

    if sys.argv[1] == 'transform_mts_to_ucr_format':
        transform_mts_to_ucr_format()

    elif sys.argv[1] == 'create_files_for_hugobot':
        config.set_path_transformations()
        create_files()

    elif sys.argv[1] == 'create_files':
        # create_files MTS mcdcnn True td4c-cosine False 1
        config.set_classifier(sys.argv[3])
        config.set_afterTA(sys.argv[4])
        config.set_method(sys.argv[5])
        config.set_combination(sys.argv[6])
        config.set_transformation(sys.argv[7])
        config.set_path_transformations()

        if config.afterTA:
            run_cpu()
