class ConfigClass:
    def __init__(self):
        self.path = "C:\\Users\\Shaha\\Desktop\\"
        self.mts_path = self.path + "mtsdata\\archives\\mts_archive\\"
        self.ucr_path = self.path + "UCRArchive_2018\\archives\\UCRArchive_2018\\"
        self.properties_path = "C:\\Users\\Shaha\\Desktop\\TA\\TEST"

        self.method = ["gradient", "equal-frequency", "equal-width", "sax", "td4c-skl", "td4c-entropy", "td4c-entropy-ig",
                       "td4c-cosine", "td4c-diffsum", "td4c-diffmax"]

        self.nb_bin = [2, 3, 5, 10]
        self.paa_window_size = [1, 2, 5, 7]
        self.std_coefficient = [-1, 2, 3]
        self.max_gap = [-1, 2, 5]

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

    def get_nb_bin(self):
        return self.nb_bin


