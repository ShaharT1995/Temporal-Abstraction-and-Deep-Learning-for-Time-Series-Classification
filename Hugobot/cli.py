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


def run_cli(config, prop_path, max_gap):
    ds_list = config.UNIVARIATE_DATASET_NAMES_2018 if config.archive == "UCR" else config.MTS_DATASET_NAMES

    for dataset_name in ds_list:
        print("\t" + dataset_name + ":")

        for file_type in ["train", "test"]:
            raw_data_path = config.path_transformation1 + dataset_name + "/transformation1_" + file_type + ".csv"

            output_folder = prop_path + dataset_name + "/" + file_type + "/"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            print("\t\t" + file_type)

            if config.method == "gradient":
                cli(['temporal-abstraction',
                     f'{raw_data_path}',  # 'Path to data set file'
                     output_folder,  # 'Path to output dir'
                     'per-property',  # per-property
                     '-s',  # -s (when using Gradient or KnowledgeBased)
                     prop_path + dataset_name+ "//gkb.csv",  # 'Path to states file' (when using Gradient or KnowledgeBased)
                     prop_path + dataset_name+  "//pp.csv",  # 'Path to pre-processing file'
                     prop_path + dataset_name+  "//ta.csv"],  # 'Path to Temporal Abstraction file'
                    standalone_mode=False)

            # Method is not gradient
            else:
                if file_type == "train":
                    cli(['temporal-abstraction',
                         f'{raw_data_path}',  # 'Path to data set file'
                         output_folder,  # 'Path to output dir'
                         'per-property',  # per-property
                         prop_path + dataset_name+ "//pp.csv",  # 'Path to pre-processing file'
                         prop_path + dataset_name+ "//ta.csv"],  # 'Path to Temporal Abstraction file'
                        standalone_mode=False)

                # Test
                else:
                    train_gkb_path = prop_path + dataset_name + "//train//states.csv"
                    df = pd.read_csv(train_gkb_path)
                    df["Method"] = "knowledge-based"

                    test_gkb_path = prop_path + dataset_name + "//test//states.csv"
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
            # else:
            #     print("\tHugobot step already done!")
        print("")
