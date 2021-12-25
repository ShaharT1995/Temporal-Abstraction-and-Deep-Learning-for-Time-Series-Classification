import os

import click
import sys
import pandas as pd

from user_interface.results_union import results_union
from user_interface.temporal_abstraction import temporal_abstraction

TEMPORAL_ABSTRACTION_PROJECT_PATH = "/"
sys.path.append(TEMPORAL_ABSTRACTION_PROJECT_PATH)


@click.group()
def cli():
    pass


cli.add_command(temporal_abstraction)
cli.add_command(results_union)

if __name__ == '__main__':
    pass


def run_cli(prop_path, data_path, ds_dict, dataset_type, method, max_gap):
    data_types = ["train", "test"]
    for dataset_name in ds_dict:
        print("\t" + dataset_name + ":")

        for file_type in data_types:
            if dataset_type == "mtsdata":
                raw_data_path = data_path + dataset_name + "/M-transformation1_" + file_type + ".csv"
            else:
                raw_data_path = data_path + dataset_name + "/" + dataset_name + "_U-transformation1_" + \
                                file_type.upper() + ".csv"
            output_folder = data_path + dataset_name + "/output/" + file_type + "/"

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            print("\t\t" + file_type)

            pp_path = prop_path + "/pp.csv"
            ta_path = prop_path + "/ta.csv"

            if method == "gradient":
                gkb_path = prop_path + "/gkb.csv"

                cli(['temporal-abstraction',
                     f'{raw_data_path}',  # 'Path to data set file'
                     output_folder,  # 'Path to output dir'
                     'per-property',  # per-property
                     '-s',  # -s (when using Gradient or KnowledgeBased)
                     gkb_path,  # 'Path to states file' (when using Gradient or KnowledgeBased)
                     pp_path,  # 'Path to pre-processing file'
                     ta_path],  # 'Path to Temporal Abstraction file'
                    standalone_mode=False)

            # method != gradient
            else:
                if file_type == "train":
                    cli(['temporal-abstraction',
                         f'{raw_data_path}',  # 'Path to data set file'
                         output_folder,  # 'Path to output dir'
                         'per-property',  # per-property
                         pp_path,  # 'Path to pre-processing file'
                         ta_path],  # 'Path to Temporal Abstraction file'
                        standalone_mode=False)

                # file type == test
                else:
                    train_gkb_path = data_path + dataset_name + "//output//train//states.csv"
                    df = pd.read_csv(train_gkb_path)
                    df["Method"] = "knowledge-based"

                    test_gkb_path = data_path + dataset_name + "//output//test//states.csv"
                    df.to_csv(test_gkb_path, index=False)

                    cli(['temporal-abstraction',
                         '-n',
                         '',
                         f'{raw_data_path}',
                         output_folder,
                         'per-dataset',
                         str(max_gap),
                         'knowledge-based',
                         test_gkb_path], standalone_mode=False)
        print("")
