from utils_folder.utils import open_pickle
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from utils_folder.configuration import ConfigClass
import matplotlib.font_manager as fm

config = ConfigClass()

font_path = "C:\Windows\Fonts\cambria.ttc"
my_font = fm.FontProperties(fname=font_path)

def concat_results(path_ta_dir):
    classifiers = ["cnn", "mlp", "mcdcnn", "fcn", "twiesn"]
    columns_df = ['classifier_name', 'archive_name', 'dataset_name', 'precision', 'accuracy', 'recall', 'duration',
                  "iteration", 'method', "nb bins", "paa", "std", "max gap", "gradient_window_size"]

    df = pd.DataFrame(columns=columns_df)

    for classifier in classifiers:
        path_ta_dir_tmp = path_ta_dir + classifier
        for root, dirs, files in os.walk(path_ta_dir_tmp):
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

    df["duration"] = df["duration"] / 1000

    # calculate F1_Score
    df["f_score"] = ((df["recall"] * df["precision"]) / (df["recall"] + df["precision"])) * 2

    if os.path.exists('results_mts_old.csv'):
        df.to_csv("results_mts_old.csv", index=False, mode='a', header=0)
    else:
        df.to_csv('results_mts_old.csv', index=False)


def graph_by_number_of_class(path):
    df = pd.read_csv(path, encoding="utf-8")

    df = df.groupby(["dataset_name", "nb bins"], as_index=False).agg(
        {"minus_accuracy": np.mean, "minus_f_score": np.mean})

    sns.set_theme(style="darkgrid")
    colors = ["#FF0B04", "#4374B3"]
    sns.set_palette(sns.color_palette(colors))

    g = sns.catplot(x="dataset_name", y="minus_f_score",
                    col="nb bins",
                    data=df, kind="point",
                    dodge=True,
                    sharey=False,
                    height=6, aspect=1.3, join=False)

    for axis in g.axes.flat:
        axis.tick_params(labelleft=True, labelbottom=True)

    plt.subplots_adjust(wspace=0.3, hspace=0.15)
    plt.show()
    #
    plt.savefig("Plots/" + "1.png", bbox_inches='tight')
    # plt.clf()

    print()


def find_best_params(df_after_ta, path_file_raw_data):
    df_after_ta = pd.read_csv(df_after_ta, encoding="utf-8")
    df_after_ta = df_after_ta.rename(columns={"f_score": "F_Score", "accuracy": "Accuracy",
                                              "duration": "Learning Time (ms)", "max gap": "Interpolation Gap"})

    # open raw data
    raw_data_df = pd.read_csv(path_file_raw_data, encoding="utf-8")

    raw_data_df["duration"] = raw_data_df["duration"] / 1000

    # calculate f score
    raw_data_df["f_score_b"] = ((raw_data_df["recall"] * raw_data_df["precision"]) /
                                (raw_data_df["recall"] + raw_data_df["precision"])) * 2

    merge_df = pd.merge(raw_data_df, df_after_ta, on=["dataset_name", "classifier_name", "archive_name", "iteration"])

    # rename columns
    merge_df = merge_df.rename(columns={"accuracy": "accuracy_b", "max gap": "max gap_b", "duration": "duration_b"})

    merge_df["minus_accuracy"] = merge_df["Accuracy"] - merge_df["accuracy_b"]
    merge_df["minus_f_score"] = merge_df["F_Score"] - merge_df["f_score_b"]

    merge_df["avg"] = (merge_df["minus_accuracy"] + merge_df["minus_f_score"]) / 2

    # filter all the positive rows
    # merge_df = merge_df.loc[(merge_df.minus_accuracy > 0) & (merge_df.minus_f_score > 0)]

    df = merge_df.groupby(["nb bins", "std", "Interpolation Gap", "paa", "gradient_window_size", "method",
                           "archive_name", "dataset_name"], as_index=False).agg({"minus_accuracy": np.mean,
                                                                                 "minus_f_score": np.mean})

    df.to_csv('BestParams.csv', index=False)


def graphs_for_all_datasets(df_after_ta, path_file_raw_data, filter_data=False, filter_specific_dataset=False,
                            paa="None", nb_bins="None", max_gap="None", gradient_window_size="None", method="None"):
    df_after_ta = pd.read_csv(df_after_ta, encoding="utf-8")

    if filter_data:
        df_after_ta = df_after_ta.loc[(df_after_ta["paa"] == paa) &
                                      (df_after_ta["nb bins"] == nb_bins) &
                                      (df_after_ta["max gap"] == max_gap) &
                                      (df_after_ta["gradient_window_size"] == str(gradient_window_size)) &
                                      (df_after_ta["method"] == method)]

    if filter_specific_dataset:
        df_after_ta = df_after_ta.loc[(df_after_ta["dataset_name"] == filter_specific_dataset)]

    df_after_ta = df_after_ta.rename(columns={"f_score": "F_Score", "accuracy": "Accuracy",
                                              "duration": "Learning Time (ms)", "max gap": "Interpolation Gap"})
    # open raw data
    raw_data_df = pd.read_csv(path_file_raw_data, encoding="utf-8")

    raw_data_df["duration"] = raw_data_df["duration"] / 1000

    # calculate f score
    raw_data_df["f_score_b"] = ((raw_data_df["recall"] * raw_data_df["precision"]) /
                                (raw_data_df["recall"] + raw_data_df["precision"])) * 2

    merge_df = pd.merge(raw_data_df, df_after_ta, on=["dataset_name", "classifier_name", "archive_name", "iteration"])

    # rename columns
    merge_df = merge_df.rename(columns={"accuracy": "accuracy_b", "max gap": "max gap_b", "duration": "duration_b"})

    melt_df_after = pd.melt(merge_df, id_vars=["dataset_name", "classifier_name"],
                            value_vars=['F_Score', "Accuracy"])

    merge_df.drop(['F_Score', "Accuracy", "Learning Time (ms)", "Interpolation Gap"], inplace=True, axis=1)

    merge_df = merge_df.rename(columns={"f_score_b": "F_Score", "accuracy_b": "Accuracy", "method": "Method",
                                        "duration_b": "Learning Time (ms)", "max gap_b": "Interpolation Gap"})

    melt_before = pd.melt(merge_df, id_vars=["dataset_name", "classifier_name"],
                          value_vars=['F_Score', "Accuracy"])

    # merge the two df
    data = pd.concat([melt_before.assign(Data='Raw Data'), melt_df_after.assign(Data='After TA')])

    data = data.rename(columns={"variable": "Evaluation Metric", "dataset_name": "Dataset Name"})

    data.loc[data.classifier_name == "fcn", "classifier_name"] = "FCN"
    data.loc[data.classifier_name == "cnn", "classifier_name"] = "CNN"
    data.loc[data.classifier_name == "mlp", "classifier_name"] = "MLP"
    data.loc[data.classifier_name == "mcdcnn", "classifier_name"] = "MCDCNN"
    data.loc[data.classifier_name == "twiesn", "classifier_name"] = "TWIESN"

    sns.set_theme(style="darkgrid")
    sns.set(font="cambria")
    colors = ["#FF0B04", "#4374B3"]
    sns.set_palette(sns.color_palette(colors))

    g = sns.catplot(x="classifier_name", y="value",
                    hue="Data", col="Evaluation Metric", row="Dataset Name",
                    data=data, kind="point",
                    sharey=False,
                    height=6, aspect=1.3, join=False)

    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            if j == 0:
                g.axes[i, j].set_ylabel('F1 Score')
            else:
                g.axes[i, j].set_ylabel('Accuracy')

            axis = g.axes[i, j]

            axis.set_xlabel('Classifier')

            axis.set_title(axis.get_title().split(" ")[3])
            axis.tick_params(labelleft=True, labelbottom=True)
            axis.set_xlabel("Classifier", fontdict={'weight': 'bold'})
            axis.set_ylabel(axis.get_ylabel(), fontdict={'weight': 'bold'})
            plt.setp(axis.get_xticklabels(), rotation=30)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

    file_name = "fig method=" + method + "_paa=" + str(paa) + "_bins=" + str(nb_bins) + "_maxGap=" + str(max_gap) + \
                "gradientWindows=" + str(gradient_window_size)
    plt.savefig("MTS_Plots/" + file_name, bbox_inches='tight')


def graph_mean_values_for_abstraction_method(df_after_ta):
    df_after_ta = pd.read_csv(df_after_ta, encoding="utf-8")

    df_after_ta = df_after_ta.rename(columns={"f_score": "F_Score", "accuracy": "Accuracy",
                                              "duration": "Learning Time (ms)", "max gap": "Interpolation Gap",
                                              "nb bins": "Number of symbols", "method": "Method"})

    melt_df_after = pd.melt(df_after_ta, id_vars=["Method"],
                            value_vars=['F_Score', "Accuracy"])

    melt_df_after = melt_df_after.rename(columns={"variable": "Evaluation Metric"})
    melt_df_after.loc[melt_df_after.Method == "gradient", "Method"] = "Gradient"
    melt_df_after.loc[melt_df_after.Method == "equal-frequency", "Method"] = "EFD"
    melt_df_after.loc[melt_df_after.Method == "equal-width", "Method"] = "EWD"
    melt_df_after.loc[melt_df_after.Method == "sax", "Method"] = "SAX"
    melt_df_after.loc[melt_df_after.Method == "td4c-cosine", "Method"] = "TD4C - Cos"

    sns.set_theme(style="darkgrid")
    sns.set(font="cambria")
    colors = ["#FF0B04", "#4374B3"]
    sns.set_palette(sns.color_palette(colors))

    g = sns.catplot(x="Method", y="value",
                    col="Evaluation Metric",
                    data=melt_df_after, kind="point",
                    dodge=True,
                    sharey=False,
                    height=6, aspect=1.3, join=False)

    g.axes[0, 0].set_ylabel('F1 Score')
    g.axes[0, 1].set_ylabel('Accuracy')

    for axis in g.axes.flat:
        axis.set_title("")
        axis.set_xlabel(axis.get_xlabel(), fontdict={'weight': 'bold'})
        axis.set_ylabel(axis.get_ylabel(), fontdict={'weight': 'bold'})
        plt.setp(axis.get_xticklabels(), rotation=30)
        axis.tick_params(labelleft=True, labelbottom=True)

    plt.subplots_adjust(wspace=0.3, hspace=0.15)
    plt.show()

    plt.savefig("Plots/12", bbox_inches='tight')


def graph_mean_values(df_after_ta, param, filter_data=False, paa="None", nb_bins="None", max_gap="None",
                      gradient_window_size="None", method="None"):
    df_after_ta = pd.read_csv(df_after_ta, encoding="utf-8")

    if filter_data:
        df_after_ta = df_after_ta.loc[(df_after_ta.paa == paa) &
                                      (df_after_ta["nb bins"] == nb_bins) &
                                      (df_after_ta["max gap"] == max_gap) &
                                      (df_after_ta["gradient_window_size"] == str(gradient_window_size)) &
                                      (df_after_ta["method"] == method)]

    df_after_ta = df_after_ta.rename(columns={"f_score": "F_Score", "accuracy": "Accuracy",
                                              "duration": "Learning Time (ms)", "max gap": "Interpolation Gap",
                                              "nb bins": "Number of symbols", "method": "Method"})

    melt_df_after = pd.melt(df_after_ta, id_vars=["Method", param],
                            value_vars=['F_Score', "Accuracy"])

    melt_df_after = melt_df_after.rename(columns={"variable": "Evaluation Metric"})
    melt_df_after.loc[melt_df_after.Method == "gradient", "Method"] = "GRAD"
    melt_df_after.loc[melt_df_after.Method == "equal-frequency", "Method"] = "EFD"
    melt_df_after.loc[melt_df_after.Method == "equal-width", "Method"] = "EWD"
    melt_df_after.loc[melt_df_after.Method == "sax", "Method"] = "SAX"
    melt_df_after.loc[melt_df_after.Method == "td4c-cosine", "Method"] = "TD4C - Cos"

    sns.set_theme(style="darkgrid")
    sns.set(font="cambria")
    colors = ["#FF0B04", "#4374B3"]
    sns.set_palette(sns.color_palette(colors))

    g = sns.catplot(x="Method", y="value",
                    hue=param, col="Evaluation Metric",
                    data=melt_df_after, kind="point",
                    sharey=False,
                    height=6, aspect=1.3, join=False)

    g.axes[0, 0].set_ylabel('F1 Score')
    g.axes[0, 1].set_ylabel('Accuracy')

    for axis in g.axes.flat:
        axis.set_title("")
        axis.set_xlabel(axis.get_xlabel(), fontdict={'weight': 'bold'})
        axis.set_ylabel(axis.get_ylabel(), fontdict={'weight': 'bold'})
        plt.setp(axis.get_xticklabels(), rotation=30)
        axis.tick_params(labelleft=True, labelbottom=True)

    plt.subplots_adjust(wspace=0.3, hspace=0.15)
    plt.show()

    file_name = "fig method=" + method + "_paa=" + str(paa) + "_bins=" + str(nb_bins) + "_maxGap=" + str(max_gap) + \
                "gradientWindows=" + str(gradient_window_size)
    plt.savefig("Plots/" + file_name, bbox_inches='tight')


def create_all_graphs():
    # after_ta_folder_path = "C:\\Users\\Shaha\\Desktop\\ProjectResult\\New2\\Results_11.02\\ResultsAfterTA\\"
    # after_ta_folder_path = "C:\\Users\\Shaha\\Desktop\\ProjectResult\\Results\\ResultsAfterTA\\"
    # raw_data_path = "C:\\Users\\Shaha\\Desktop\\raw_data_results.csv"

    after_ta_folder_path = "C:\\Users\\Shaha\\Desktop\\ProjectResult\\MTS_New\\MTS_New_PAA\\Results_MTS_NEW_12.02\\ResultsAfterTA\\"
    raw_data_path = "C:\\Users\\Shaha\\Desktop\\results_raw_mts.csv"

    # concat_results(after_ta_folder_path)

    # find_best_params("results_ucr.csv", raw_data_path)

    # param = paa / Interpolation Gap / Number of symbols / gradient_window_size

    # graph_mean_values("results_new.csv",
    #                   filter_data=True,
    #                   param=paa
    #                   paa=1,
    #                   nb_bins=10,
    #                   gradient_window_size=20,
    #                   method="gradient",
    #                   max_gap=1)

    # METHODS = ['sax', 'equal-frequency', 'equal-width']
    #
    # for method in METHODS:
    #     for nb_bin in config.get_nb_bin():
    #         for paa in config.get_paa_window_size():
    #             for std in config.get_std_coefficient():
    #                 for max_gap in config.get_max_gap():
    #                     if method == "gradient":
    #                         for gradient_window in config.get_gradient_window_size():
    #                             params = "res_" + str(method) + "" + str(nb_bin) + "" + str(paa) + "_" + str(std) \
    #                                      + "" + str(max_gap) + "" + str(gradient_window)
    #
    #                             print(params)
    #
    #                             graphs_for_all_datasets(df_after_ta="results_mts.csv",
    #                                                     path_file_raw_data=raw_data_path,
    #                                                     filter_data=True,
    #                                                     paa=paa,
    #                                                     nb_bins=nb_bin,
    #                                                     max_gap=max_gap,
    #                                                     gradient_window_size=gradient_window,
    #                                                     method=method)
    #
    #                     else:
    #                         params = "res_" + str(method) + "" + str(nb_bin) + "" + str(paa) + "_" + str(std) \
    #                                  + "" + str(max_gap) + "None"
    #
    #                         print(params)
    #
    #                         graphs_for_all_datasets(df_after_ta="results_mts.csv",
    #                                                 path_file_raw_data=raw_data_path,
    #                                                 filter_data=True,
    #                                                 paa=paa,
    #                                                 nb_bins=nb_bin,
    #                                                 max_gap=max_gap,
    #                                                 gradient_window_size=None,
    #                                                 method=method)

    # graphs_for_all_datasets(df_after_ta="results_ucr.csv",
    #                         path_file_raw_data=raw_data_path,
    #                         filter_data=True,
    #                         paa=1,
    #                         nb_bins=2,
    #                         max_gap=1,
    #                         gradient_window_size=None,
    #                         method="sax")

    # Number of symbols / paa / Interpolation Gap
    # graph_mean_values(df_after_ta="results_mts.csv",
    #                   param="Interpolation Gap",
    #                   filter_data=False)

    graphs_for_all_datasets(df_after_ta="results_mts.csv",
                            path_file_raw_data=raw_data_path)

    # graph_by_number_of_class("BestParams.csv")
    # graph_mean_values_for_abstraction_method("results_mts.csv")


create_all_graphs()
