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


def run():
    prop_path = config.path_files_for_TA + config.archive + "//" + config.classifier + "//" + config.method + "//"
    create_directory(prop_path)

    # Make the 3 files - gkb.csv, ta.csv and ppa.csv
    print("Step 2: make the gkb.csv, ta.csv and ppa.csv \n")

    if not check_pickle_exists("running_dict" + config.archive):
        write_pickle("running_dict" + config.archive, {})

    running_dict = open_pickle("running_dict" + config.archive)

    for nb_bin in config.nb_bin:
        for std in config.std_coefficient:
            for max_gap in config.max_gap:
                if config.method == "gradient":
                    for gradient_window in config.gradient_window_size:
                        running_dict = execute_running(config, prop_path, running_dict, max_gap, config.method, nb_bin,
                                                       config.paa_window_size, std, gradient_window)
                else:
                    running_dict = execute_running(config, prop_path, running_dict, max_gap, config.method, nb_bin,
                                                   config.paa_window_size, std)
    print("Done")


def execute_running(config, prop_path, running_dict, max_gap, method, nb_bin, paa, std, gradient_window=None):
    # todo - Add gradient to the print
    print("-------------------------------------------------------------------------------------")
    print("Method: " + method + ", Bins: " + str(nb_bin) + ", PAA: " + str(paa) + ", STD: " +
          str(std) + ", Max_Gap: " + str(max_gap))
    print("------------------------------------------------------------------------------------- \n")

    key = (config.archive, config.classifier, method, nb_bin, paa, std, max_gap, gradient_window)

    if key in running_dict:
        print("Already Done! \n")

        return running_dict

    else:
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

        # Make the second temporal abstraction -> hugobot output files to original format
        print("Step 4: transformation 2")
        new_ucr_files(config, prop_path) if config.archive == "UCR" else new_mts_files(config, prop_path)

        print("Step 5: Run all:")
        params = "res_" + str(method) + "_" + str(nb_bin) + "_" + str(paa) + "_" + str(std) \
                        + "_" + str(max_gap) + "_" + str(gradient_window)
        run_models.run_all(config, params)
        print("")

        print("Step 6: Generate Results to CSV")
        generate_results_csv(config, params)

        running_dict[key] = True
        write_pickle("running_dict" + config.archive, running_dict)
        return running_dict


if __name__ == '__main__':
    import sys
    import run_models

    sys.path.insert(0, '/sise/robertmo-group/TA-DL-TSC/Project/')

    from utils_folder.configuration import ConfigClass
    from utils_folder.utils import generate_results_csv, write_pickle, open_pickle, transform_mts_to_ucr_format, \
        create_directory, check_pickle_exists

    config = ConfigClass()
    config.set_archive(sys.argv[2])

    from temporal_abstraction_f.multivariate_ta_1 import MultivariateTA1
    from temporal_abstraction_f.univariate_ta_1 import UnivariateTA1
    from temporal_abstraction_f.set_parameters import create_three_files
    from temporal_abstraction_f.multivariate_ta_2 import new_mts_files
    from temporal_abstraction_f.univariate_ta_2 import new_ucr_files

    sys.path.insert(0, '/sise/robertmo-group/TA-DL-TSC/Project/Hugobot')
    from Hugobot.cli import run_cli

    if sys.argv[1] == 'create_files_for_hugobot':
        config.set_path_transformations()
        create_files()

    if sys.argv[1] == 'run_all':
        config.set_classifier(sys.argv[3])
        config.set_afterTA(sys.argv[4])
        config.set_method(sys.argv[5])
        config.set_path_transformations()

        run() if config.afterTA else run_models.run_all(config, "RawData")

    elif sys.argv[1] == 'transform_mts_to_ucr_format':
        transform_mts_to_ucr_format()

    elif sys.argv[1] == 'generate_results_csv':
        config.set_classifier(sys.argv[3])
        config.set_afterTA(sys.argv[4])
        config.set_method(sys.argv[5])
        config.set_path_transformations()

        generate_results_csv(config, "RawData")
