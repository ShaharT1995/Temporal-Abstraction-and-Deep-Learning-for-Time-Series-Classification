class ConfigClass:
    def __init__(self):
        self.path = "/sise/robertmo-group/TA-DL-TSC/"
        self.mts_path = self.path + "mtsdata//archives//mts_archive/"
        self.ucr_path = self.path + "UCRArchive_2018//archives//UCRArchive_2018/"
        self.properties_path = self.path + "SyncProject//TA//"

        # self.method = ["gradient", "equal-frequency", "equal-width", "sax", "td4c-skl", "td4c-entropy",
        #                "td4c-entropy-ig", "td4c-cosine", "td4c-diffsum", "td4c-diffmax"]
        #
        # self.nb_bin = [5]
        # self.std_coefficient = [-1]
        #
        # self.paa_window_size = [1]
        #
        self.gradient_window_size = [2]
        #
        # # interpolation_gap
        # self.max_gap = [5]

        self.method = ["equal-frequency"]
        self.nb_bin = [3]
        self.paa_window_size = [1]
        self.std_coefficient = [-1]
        self.max_gap = [1]

        # self.nb_bin = [2, 3, 5, 10, 25]
        # self.std_coefficient = [-1]
        #
        # self.paa_window_size = {1: [1, 2],
        #                         2: [1, 2, 5],
        #                         3: [1, 2, 5, 10],
        #                         4: [1, 2, 5, 10, 20],
        #                         5: [1, 2, 5, 10, 20, 50],
        #                         6: [1, 2, 5, 10, 20, 50, 100]}
        #
        # self.gradient_window_size = {1: [-1, 2],
        #                              2: [-1, 2, 5],
        #                              3: [-1, 2, 5, 10],
        #                              4: [-1, 2, 5, 10, 20],
        #                              5: [-1, 2, 5, 10, 20, 50],
        #                              6: [-1, 2, 5, 10, 20, 50, 100]}
        #
        # # interpolation_gap
        # self.max_gap = {1: [-1, 1, 2],
        #                 2: [-1, 2, 5],
        #                 3: [-1, 2, 5, 10],
        #                 4: [-1, 2, 5, 10, 20],
        #                 5: [-1, 2, 5, 10, 20, 50],
        #                 6: [-1, 2, 5, 10, 20, 50, 100]}

    def get_path(self):
        return self.path

    def get_mts_path(self):
        return self.mts_path

    def get_ucr_path(self):
        return self.ucr_path

    def get_prop_path(self):
        return self.properties_path

    def get_method(self):
        return self.method

    def get_max_gap(self):
        return self.max_gap

    def get_std_coefficient(self):
        return self.std_coefficient

    def get_paa_window_size(self):
        return self.paa_window_size

    def get_gradient_window_size(self):
        return self.gradient_window_size

    def get_nb_bin(self):
        return self.nb_bin
