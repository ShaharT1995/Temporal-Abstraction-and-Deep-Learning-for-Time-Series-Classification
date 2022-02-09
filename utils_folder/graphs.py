import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


def open_pickle(name):
    file = open("C:\\Users\\Shaha\\Desktop\\" + name + ".pkl", "rb")
    data = pickle.load(file)
    return data


def concat_results(path_ta_dir):
    classifiers = ["cnn", "mlp", "mcdcnn", "fcn", "twiesn"]

    columns_df = ['classifier_name', 'archive_name', 'dataset_name', 'precision', 'accuracy', 'recall', 'duration',
                  "iteration", 'method', "nb bins", "paa", "std", "max gap", "gradient_window_size"]

    df = pd.DataFrame(columns=columns_df)
    for classifier in classifiers:
        path_ta_dir += classifier
        for root, dirs, files in os.walk(path_ta_dir):
            for file in files:
                if file.endswith(".csv"):
                    arguments = re.split('[_.]', file)
                    res_ta_data = pd.read_csv(root + "//" + file, sep=',', header=0, encoding="utf-8")
                    res_ta_data["method"] = arguments[1]
                    res_ta_data["nb bins"] = arguments[2]
                    res_ta_data["paa"] = arguments[3]
                    res_ta_data["std"] = arguments[4]
                    res_ta_data["max gap"] = arguments[5]
                    res_ta_data["gradient_window_size"] = arguments[6]
                    df = pd.concat([df, res_ta_data])

    if os.path.exists('results_new.csv'):
        df.to_csv("results_new.csv", index=False, mode='a', header=0)
    else:
        df.to_csv('results_new.csv', index=False)


def set_df_for_graphs_and_tables(path_file_raw_data, path_after_ta_df, param):
    # open raw data
    raw_data_df = pd.read_csv(path_file_raw_data, encoding="utf-8")

    # rename columns
    raw_data_df = raw_data_df.rename(columns={"f_score": "f_score_b", "duration": "duration_b",
                                              "accuracy": "accuracy_b", "max gap": "Interpolation Gap"})

    raw_data_df["duration_b"] = raw_data_df["duration_b"] / 1000
    # calculet f score
    raw_data_df["f_score_b"] = ((raw_data_df["recall"] * raw_data_df["precision"]) /
                                (raw_data_df["recall"] + raw_data_df["precision"])) * 2

    # Open data after TA
    data_after_ta_df = pd.read_csv(path_after_ta_df, encoding="utf-8")

    # Calculate F_Score
    data_after_ta_df["f_score"] = ((data_after_ta_df["recall"] * data_after_ta_df["precision"]) /
                                   (data_after_ta_df["recall"] + data_after_ta_df["precision"])) * 2
    data_after_ta_df["duration"] = data_after_ta_df["duration"] / 1000

    # Rename columns
    data_after_ta_df = data_after_ta_df.rename(columns={"f_score": "F_Score", "duration": "Learning Time (ms)",
                                                        "accuracy": "Accuracy", "max gap": "Interpolation Gap"})

    df = pd.merge(raw_data_df, data_after_ta_df, on=["dataset_name", "classifier_name", "archive_name", "iteration"])

    ucr_length = open_pickle("uts_length")
    mts_length = {}
    lengths = {**ucr_length, **mts_length}
    length_df = pd.DataFrame(list(lengths.items()), columns=['dataset_name', 'lengths'])

    merged = pd.merge(df, length_df, on="dataset_name")

    merged.loc[merged.lengths < 81, 'groups_lengths'] = "< 81"
    merged.loc[(merged.lengths >= 81) & (merged.lengths <= 250), 'groups_lengths'] = "81 - 250"
    merged.loc[(merged.lengths >= 251) & (merged.lengths <= 450), 'groups_lengths'] = "251 - 450"
    merged.loc[(merged.lengths >= 451) & (merged.lengths <= 700), 'groups_lengths'] = "451 - 700"
    merged.loc[(merged.lengths >= 701) & (merged.lengths <= 1000), 'groups_lengths'] = "701 - 1000"
    merged.loc[merged.lengths > 1000, 'groups_lengths'] = " > 1000"

    # res_df = merged.groupby(["archive_name", "groups_lengths", param], as_index=False).agg \
    #     ({"duration_b": np.mean, "duration": np.mean,"f_score": np.mean, "f_score_b": np.mean, "accuracy": np.mean, "accuracy_b": np.mean})
    #
    # return res_df
    return merged


def find_best_params(path):
    df = pd.read_csv(path, encoding="utf-8")

    df = df.loc[df.paa == 1]

    df["f_score"] = ((df["recall"] * df["precision"]) / (df["recall"] + df["precision"])) * 2

    df = df.groupby(["nb bins", "std", "max gap", "paa", "gradient_window_size"], as_index=False).agg \
        ({"f_score": np.mean, "duration": np.mean, "accuracy": np.mean})

    top_f_score = df.sort_values(by=['f_score'], ascending=False).head(3)
    top_accuracy = df.sort_values(by=['accuracy'], ascending=False).head(10)
    top_duration = df.sort_values(by=['duration'], ascending=True).head(10)

    merged = pd.merge(top_accuracy, top_f_score, on=["nb bins", "std", "max gap", "paa", "gradient_window_size"])
    merged = pd.merge(merged, top_duration, on=["nb bins", "std", "max gap", "paa", "gradient_window_size"])

    print()


def create_graphs(res_df, param, archive_name="UCRArchive_2018"):
    res_df = res_df.loc[(res_df.archive_name == archive_name) & (res_df.paa == 1)]
    melt_df_after = pd.melt(res_df, id_vars=['method', "archive_name", param],
                            value_vars=['F_Score', "Accuracy", "Learning Time (ms)"])

    res_df.drop(['F_Score', "Accuracy", "Learning Time (ms)"], inplace=True, axis=1)

    res_df = res_df.rename(columns={"f_score_b": "F_Score", "accuracy_b": "Accuracy",
                                    "duration_b": "Learning Time (ms)"})

    melt_before = pd.melt(res_df, id_vars=['method', "archive_name", param],
                          value_vars=['F_Score', "Accuracy", "Learning Time (ms)"])

    data = pd.concat([melt_before.assign(Data='Raw Data'),
                      melt_df_after.assign(Data='After TA')])

    sns.set_theme(style="darkgrid")
    colors = ["#FF0B04", "#4374B3"]
    sns.set_palette(sns.color_palette(colors))

    g = sns.FacetGrid(data, row=param, col="variable", hue="Data", height=6, aspect=1.2)
    g = g.map(sns.lineplot, "method", "value", ci=95, err_style="bars")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    for axis in g.axes.flat:
        axis.tick_params(labelleft=True, labelbottom=True)

    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    plt.savefig("Plots/" + param + ".png", bbox_inches='tight')
    plt.clf()


def create_all_graphs():
    path = "C:\\Users\\Shaha\\Desktop\\ResultsAfterTA\\"
    concat_results(path)

    find_best_params("results_new.csv")

    for param in ["nb bins", "Interpolation Gap"]:
        res_df = set_df_for_graphs_and_tables("C:\\Users\\Shaha\\Desktop\\raw_data_results.csv",
                                              "results_new.csv",
                                              param)
        create_graphs(res_df, param)


create_all_graphs()
