import os
import re

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt

from utils_folder.configuration import ConfigClass
from utils_folder.utils import open_pickle

config = ConfigClass()

font_path = "/sise/robertmo-group/TA-DL-TSC/Cambria/CAMBRIA.TTC"
my_font = fm.FontProperties(fname=font_path)

font_path = "/sise/robertmo-group/TA-DL-TSC/Cambria/CAMBRIAB.TTF"
my_font_bold = fm.FontProperties(fname=font_path)


# Creates a unified file of all results for all combinations
def concat_results(path_ta_dir, normalization=False, raw_data=False, type="UCR"):
    classifiers = ['fcn', 'resnet', 'inception', 'mcdcnn', 'mlstm_fcn', 'cnn', 'mlp']
    methods = ['sax', 'equal-frequency', 'equal-width']
    bins = ["3", "5", "10", "20"]

    columns_df = ['classifier_name', 'archive_name', 'dataset_name', 'Precision', 'Accuracy', 'Recall', 'MCC',
                  'Cohen Kappa', 'Learning Time', 'Predicting Time', 'F1 Score Macro', 'F1 Score Micro',
                  'F1 Score Weighted', 'Balanced Accuracy', 'AUC', "iteration"]

    path = "Reports/"
    path += "With ZNorm/" if normalization else "Without ZNorm/"
    path += type + "/Per Entity/"

    output_file_name = path + "RawData.csv"
    if not raw_data:
        columns_df += ['method', "nb bins", "paa", "std", "max gap", "gradient_window_size"]
        output_file_name = path + 'TA.csv'

    df = pd.DataFrame(columns=columns_df)
    for classifier in classifiers:
        path_ta_dir_tmp = path_ta_dir + classifier
        for root, dirs, files in os.walk(path_ta_dir_tmp):
            for method in dirs:
                if method not in methods:
                    continue

                files_path = path_ta_dir_tmp + "/" + method
                for root, dirs, files in os.walk(files_path):
                    for file in files:
                        if file.endswith(".csv"):
                            arguments = re.split('[_.]', file)

                            if not raw_data and (arguments[8] == "True" or arguments[2] not in bins or
                                                 arguments[7] == "3" or arguments[7] == "1" or arguments[9] == "False"):
                                continue

                            res_ta_data = pd.read_csv(root + "//" + file, sep=',', header=0, encoding="utf-8")

                            if not raw_data:
                                new_method = method + " with Gradient" if arguments[8] == "True" else method
                                new_method = new_method + " with Per Entity" if arguments[9] == "True" else new_method
                                res_ta_data["method"] = new_method
                                res_ta_data["nb bins"] = arguments[2]
                                res_ta_data["paa"] = arguments[3]
                                res_ta_data["std"] = arguments[4]
                                res_ta_data["max gap"] = arguments[5]
                                res_ta_data["gradient_window_size"] = arguments[6]

                                if arguments[7] == "1":
                                    transformation_name = "Discrete"
                                elif arguments[7] == "2":
                                    transformation_name = "Symbol One-Hot"
                                else:
                                    transformation_name = "Endpoint One-Hot"

                                res_ta_data["transformation_type"] = transformation_name

                            df = pd.concat([df, res_ta_data])

    df["Learning Time"] = df["Learning Time"] / 1000
    df["Predicting Time"] = df["Predicting Time"] / 1000

    df = df.drop(['precision', 'recall', 'accuracy', 'mcc', 'cohen_kappa', 'f1_score_macro',
                  'f1_score_micro', 'f1_score_weighted', 'learning_time', 'predicting_time',
                  'auc'], axis=1, errors='ignore')
    df = df.rename(columns={"AUC": "AUC - ROC", 'Accuracy': 'Balanced Accuracy'})

    df = df.replace({"classifier_name": dict(fcn="FCN", mlp="MLP", resnet="ResNet", twiesn="TWIESN", encoder="Encoder",
                                             mcdcnn="MCDCNN", cnn="Time - CNN", inception="Inception",
                                             lstm_fcn="LSTM - FCN", mlstm_fcn="MLSTM - FCN", rocket="Rocket")})

    # Rename raw data columns
    if raw_data:
        df.rename(columns=lambda x: x + "_raw_data", inplace=True)
    else:
        df = df.replace({"method": {"sax": "SAX", "td4c-cosine": "TD4C - Cosine", "gradient": "Gradient",
                                    "sax with Gradient": "SAX with Gradient", "td4c-cosine with Gradient":
                                        "TD4C - Cosine with Gradient", "sax with Per Entity": "SAX with Per Entity",
                                    "equal-frequency": "Equal-Frequency", "equal-width": "Equal-Width"}})

    df.to_csv(output_file_name, index=False)


def merge_two_df(raw_df, ta_df, metrics):
    ta_df = ta_df.groupby(["classifier_name", "dataset_name", "method", "nb bins", "transformation_type", "PAA",
                           "Interpolation Gap"]
                          , as_index=False).agg({k: np.mean for k in metrics})
    raw_df = raw_df.groupby(["classifier_name_raw_data", "dataset_name_raw_data"],
                            as_index=False).agg({k + "_raw_data": np.mean for k in metrics})

    raw_df = raw_df.rename(columns={"classifier_name_raw_data": "classifier_name",
                                    "dataset_name_raw_data": "dataset_name"})

    merged = pd.merge(raw_df, ta_df, on=["classifier_name", "dataset_name"])
    return merged


def get_best_df_after_ta(ta_df, metrics, lst_group_by, max_val=None):
    dict_ta = {k: np.mean for k in metrics}
    df = ta_df.groupby(lst_group_by, as_index=False).agg(dict_ta)

    df["avg_ta"] = df[[i for i in metrics]].mean(axis=1)

    # df = df.loc[(df["nb bins"] == 3) & (df["method"] == "GRAD")]

    # Get only the top X results
    if max_val is not None:
        df = df.nlargest(max_val, "avg_ta")

    best_params = df[lst_group_by]
    keys = list(best_params.columns.values)
    i1 = ta_df.set_index(keys).index
    i2 = best_params.set_index(keys).index
    df = ta_df[i1.isin(i2)]

    return df


def create_fig(x, y, col, data, name, x_label, y_label='', legend='', hue=None, type="UCR", order=None,
               join=False, graph_num=1, colors=1):
    graph_aspect = 1.5

    rotation = 30

    if graph_num == 5:
        rotation = 70
        metrics = ['MCC', 'Cohen Kappa', 'F1 Score Macro', 'F1 Score Micro', 'F1 Score Weighted', 'Balanced Accuracy',
                   'AUC - ROC']

    elif graph_num == 1 or graph_num == 4:
        rotation = 0
        metrics = ['Balanced Accuracy', "AUC - ROC"]
        data = data.loc[(data['Evaluation Metric'] == "AUC - ROC") | (data['Evaluation Metric'] == "Balanced Accuracy")]

    elif graph_num == 7 or graph_num == 6:
        metrics = ['Balanced Accuracy', "AUC - ROC"]
        # metrics = ['Balanced Accuracy']
        data = data.loc[(data['Evaluation Metric'] == "Balanced Accuracy") | (data['Evaluation Metric'] == "AUC - ROC")]
        graph_aspect = 2 if type == "UCR" else 2

    else:
        metrics = ['Balanced Accuracy', "AUC - ROC"]
        data = data.loc[(data['Evaluation Metric'] == "AUC - ROC") | (data['Evaluation Metric'] == "Balanced Accuracy")]

    if colors == 1:
        sns.set_palette(sns.color_palette(["#CD37CB", "#26AA1B"]))
    elif colors == 2:
        sns.set_palette(sns.color_palette(["#D422AE", "#D422AE"]))
    else:
        sns.set_palette(sns.color_palette(["#FF0B04", "#4374B3"]))

    g = sns.catplot(x=x, y=y, col=col, hue=hue, data=data, kind="point", sharey=False, height=6, aspect=graph_aspect,
                    join=join, order=order, legend=False, scale=1.3)
    v = 0

    matplotlib.font_manager._load_fontmanager(try_read_cache=False)

    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            plt.rcParams["font.family"] = "Cambria"

            axis = g.axes[i, j]
            # if i == 0 and j == 0:
            #     axis.set_ylim((0.61, 0.62))
            axis.set_xlabel(x_label, fontdict={'weight': 'bold'}, fontproperties=my_font_bold, size="20")
            axis.tick_params(labelleft=True, labelbottom=True)

            if legend != "":
                axis.legend(title=legend, prop={'family': 'Cambria', 'size': 20}, title_fontsize='20')

            axis.set_title(None)
            y_label_new = metrics[v] if y_label == '' else y_label

            plt.setp(axis.get_yticklabels(), fontproperties=my_font, fontsize="20")
            plt.setp(axis.get_xticklabels(), rotation=rotation, fontproperties=my_font, fontsize="20")

            axis.set_ylabel(y_label_new, fontproperties=my_font_bold, size="20")

            v += 1

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.show()

    output_path = "ArticleGraphs - Per Entity/" + type + "/" + name
    plt.savefig(output_path, bbox_inches='tight')


def melt_RawData_TA(df, lst, metrics):
    melt_ta = pd.melt(df, id_vars=lst, value_vars=metrics)

    df = df.drop(metrics, axis=1)
    df.columns = df.columns.str.replace('_raw_data', '')

    melt_raw = pd.melt(df, id_vars=lst, value_vars=metrics)

    # Merge the two df
    data = pd.concat([melt_raw.assign(Data='Raw Data'), melt_ta.assign(Data='TA')])

    return data


def metrics_best_combination_VS_raw_data(df, metrics, type="UCR"):
    order = ['Balanced Accuracy', 'AUC - ROC', 'F1 Micro', 'F1 Macro', 'F1 Weighted', 'MCC', "Cohen\'s Kappa"]

    df = df.rename(columns={'F1 Score Micro': 'F1 Micro', 'F1 Score Macro': 'F1 Macro',
                            'F1 Score Weighted': 'F1 Weighted', "Cohen Kappa": "Cohen\'s Kappa",
                            'F1 Score Micro_raw_data': 'F1 Micro_raw_data',
                            'F1 Score Macro_raw_data': 'F1 Macro_raw_data',
                            'F1 Score Weighted_raw_data': 'F1 Weighted_raw_data',
                            "Cohen Kappa_raw_data": "Cohen\'s Kappa_raw_data"})

    data = melt_RawData_TA(df, ["nb bins", "method", "classifier_name", "transformation_type"], order)

    data = data.rename(columns={"variable": "Evaluation Metric", "method": "Method",
                                "transformation_type": "Transformation Type"})

    create_fig(x="Evaluation Metric", y="value", col=None, data=data, name="TA vs Raw data", legend="TA vs Raw data",
               x_label='Evaluation Metric', y_label="Metric Value", hue="Data", type=type, graph_num=5, colors=3,
               order=order)


def classifiers_best_combination_VS_raw_data(df, metrics, type="UCR"):
    data = melt_RawData_TA(df, ["nb bins", "method", "transformation_type", "classifier_name"], metrics)

    data = data.rename(columns={"variable": "Evaluation Metric", "classifier_name": "Classifier"})

    order = ["MLP", "MC-DCNN", "Time - CNN", "FCN", "ResNets", "InceptionTime", "MLSTM - FCN"]

    create_fig(x="Classifier", y="value", col="Evaluation Metric", data=data, name="TA vs Raw data - Classifier",
               x_label='Deep Neural Network', legend="TA vs Raw data", hue="Data", type=type, colors=3, graph_num=6,
               order=order)


def datasets_best_combination_VS_raw_data(df, metrics, type="UCR"):
    data = melt_RawData_TA(df, ["nb bins", "method", "transformation_type", "dataset_name"], metrics)

    data = data.rename(columns={"variable": "Evaluation Metric", "dataset_name": "Dataset"})

    create_fig(x="Dataset", y="value", col="Evaluation Metric", data=data, name="TA vs Raw data - Datasets",
               x_label='Dataset', legend="TA vs Raw data", hue="Data", type=type, graph_num=7, colors=3)


def create_all_graphs(graph_numbers, create_csv=False, type="UCR"):
    config = ConfigClass()
    normalization = config.normalization

    # Run this only once
    if create_csv:
        # path_for_ta_dir = config.path + "ResultsProject/"
        # path_for_ta_dir += "RawData - With ZNorm/" + type + "/" if normalization else \
        #     "RawData - Without ZNorm/" + type + "/"
        #
        # concat_results(path_for_ta_dir, raw_data=True, normalization=normalization, type=type)

        path_for_ta_dir = config.path + "ResultsProject/AfterTA/" + type
        path_for_ta_dir += " - With ZNorm/" if normalization else " - Without ZNorm/"

        concat_results(path_for_ta_dir, normalization=normalization, type=type)

    path_after_ta = "Reports/Without ZNorm/" + type + "/Per Entity/"
    path_raw_data = "Reports/With ZNorm/" + type + "/"

    # Open after TA data
    df_after_ta = pd.read_csv(path_after_ta + "TA.csv", encoding="utf-8")

    # Open raw data
    raw_data_df = pd.read_csv(path_raw_data + "RawData.csv", encoding="utf-8")

    df_after_ta = df_after_ta.replace({"method": {"sax": "SAX", "Gradient": "GRAD", "Equal-Frequency": "EFD",
                                                  "Equal-Width": "EWD"},
                                       "classifier_name": {"MCDCNN": "MC-DCNN", "ResNet": "ResNets",
                                                           "Inception": "InceptionTime"}})

    raw_data_df = raw_data_df.replace({"classifier_name_raw_data": {"MCDCNN": "MC-DCNN", "ResNet": "ResNets",
                                                                    "Inception": "InceptionTime"}})

    metrics = ['MCC', 'Cohen Kappa', 'F1 Score Macro', 'F1 Score Micro', 'F1 Score Weighted', 'Balanced Accuracy',
               'AUC - ROC']

    df_after_ta = df_after_ta.rename(columns={"max gap": "Interpolation Gap", "paa": "PAA"})

    df = merge_two_df(raw_data_df, df_after_ta, metrics)

    for num in graph_numbers:
        # PAA Graph
        if num == 1:
            order = [1, 2, 5]

            melt_df = pd.melt(df_after_ta, id_vars=["nb bins", "PAA", "method", "classifier_name",
                                                    "transformation_type"], value_vars=metrics)

            melt_df = melt_df.rename(columns={"variable": "Evaluation Metric"})
            create_fig(x="PAA", y="value", col="Evaluation Metric", data=melt_df, name="PAA",
                       x_label="PAA Window Size", type=type, colors=1, order=order, graph_num=9, join=True)

        # Interpolation gap Graph
        elif num == 2:
            order = [1, 2, 3]

            melt_df = pd.melt(df_after_ta, id_vars=["nb bins", "Interpolation Gap", "method", "classifier_name",
                                                    "transformation_type"], value_vars=metrics)

            melt_df = melt_df.rename(columns={"variable": "Evaluation Metric"})
            create_fig(x="Interpolation Gap", y="value", col="Evaluation Metric", data=melt_df,
                       name="Interpolation Gap", x_label="Interpolation Gap Window Size", type=type, colors=1,
                       order=order, graph_num=9, join=True)

        # Graphs of the methods
        elif num == 3:
            df = df_after_ta.loc[(df_after_ta["PAA"] == 1) & (df_after_ta["Interpolation Gap"] == 1)]

            order = ["EWD", "EFD", "SAX"]

            melt_df = pd.melt(df, id_vars=["nb bins", "method", "classifier_name", "transformation_type"],
                              value_vars=metrics)

            melt_df = melt_df.rename(columns={"variable": "Evaluation Metric", "method": "Method"})
            create_fig(x="Method", y="value", col="Evaluation Metric", data=melt_df, name="Method",
                       x_label='Method', hue="nb bins", legend="Number of Symbols", type=type, colors=1,
                       order=order)

        # All the metric - Raw Data VS. best combination
        elif num == 4:
            df = df.loc[(df["PAA"] == 1) & (df["Interpolation Gap"] == 1)]

            metrics_graph_5 = ['Balanced Accuracy', 'AUC - ROC']
            df = get_best_df_after_ta(df, metrics_graph_5, ["nb bins", "method", "transformation_type"],
                                      max_val=1)
            metrics_best_combination_VS_raw_data(df, metrics, type=type)

        # Classifiers - Raw Data VS. best combination
        elif num == 5:
            df = df.loc[(df["PAA"] == 1) & (df["Interpolation Gap"] == 1)]

            df = get_best_df_after_ta(df, metrics, ["nb bins", "method", "transformation_type"], max_val=1)
            classifiers_best_combination_VS_raw_data(df, metrics, type=type)


if __name__ == '__main__':
    create_all_graphs([4, 5], create_csv=False, type="MTS")
