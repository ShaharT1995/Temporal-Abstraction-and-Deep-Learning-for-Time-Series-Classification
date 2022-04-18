def run():
    if not check_pickle_exists("running_dict" + config.archive):
        write_pickle("running_dict" + config.archive, {})

    running_dict = open_pickle("running_dict" + config.archive)

    for nb_bin in config.nb_bin:
        for std in config.std_coefficient:
            for max_gap in config.max_gap:
                if config.method == "gradient":
                    for gradient_window in config.gradient_window_size:
                        running_dict = execute_running(config, running_dict, max_gap, config.method, nb_bin,
                                                       config.paa_window_size, std, gradient_window)
                else:
                    running_dict = execute_running(config, running_dict, max_gap, config.method, nb_bin,
                                                   config.paa_window_size, std)
    print("Done")


def execute_running(config, running_dict, max_gap, method, nb_bin, paa, std, gradient_window=None):
    print("-----------------------------------------------------------------------------------------------------")
    print("Classifier: " + config.classifier + ", Method: " + method + ", Bins: " + str(nb_bin) + " Combination: " +
          str(config.combination) + ", Transformation Number: " + str(config.transformation_number) + ", PerEntity: "
          + str(config.perEntity))
    print("----------------------------------------------------------------------------------------------------- \n")

    key = (config.archive, config.classifier, method, nb_bin, paa, std, max_gap, gradient_window,
           config.transformation_number, config.combination, config.perEntity)
    # set number of bins in the path
    config.set_path_transformations_2(nb_bin)

    """
    if key in running_dict:
        print("Already Done! \n")
    

        return running_dict
    """
    if False:
        print()
    else:
        print("Step 5: Run all:")
        params = "res_" + str(method) + "_" + str(nb_bin) + "_" + str(paa) + "_" + str(std) \
                        + "_" + str(max_gap) + "_" + str(gradient_window) + "_" + str(config.transformation_number)\
                        + "_" + str(config.combination) + "_" + str(config.perEntity)

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
    from utils_folder.utils import generate_results_csv, write_pickle, open_pickle, check_pickle_exists

    config = ConfigClass()
    config.set_archive(sys.argv[2])

    if sys.argv[1] == 'run_all':
        # run_all UCR mcdcnn True sax
        config.set_classifier(sys.argv[3])
        config.set_afterTA(sys.argv[4])
        config.set_method(sys.argv[5])
        config.set_combination(sys.argv[6])
        config.set_transformation(sys.argv[7])
        config.set_path_transformations()

        run() if config.afterTA else run_models.run_all(config, "RawData")

        if not config.afterTA:
            print("Step 6: Generate Results to CSV")
            generate_results_csv(config, "RawData")

    elif sys.argv[1] == 'generate_results_csv':
        config.set_classifier(sys.argv[3])
        config.set_afterTA(sys.argv[4])
        config.set_method(sys.argv[5])
        config.set_path_transformations()

        generate_results_csv(config, "RawData")
