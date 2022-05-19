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
def concat_results(path_ta_dir, raw_data=False, type="UCR"):
    classifiers = ["cnn", "mlp", "mcdcnn", "fcn", "twiesn", "encoder", "inception", "lstm_fcn", "mlstm_fcn", "rocket"]

    columns_df = ['classifier_name', 'archive_name', 'dataset_name', 'Precision', 'Accuracy', 'Recall', 'MCC',
                  'Cohen Kappa', 'Learning Time', 'Predicting Time', 'F1 Score Macro', 'F1 Score Micro',
                  'F1 Score Weighted', 'Balanced Accuracy', 'AUC', "iteration"]

    output_file_name = "Raw_data_results_" + type + "_new.csv"
    if not raw_data:
        columns_df += ['method', "nb bins", "paa", "std", "max gap", "gradient_window_size"]
        output_file_name = 'results_of_all_combinations_' + type + '_new.csv'

    df = pd.DataFrame(columns=columns_df)
    for classifier in classifiers:
        path_ta_dir_tmp = path_ta_dir + classifier
        for root, dirs, files in os.walk(path_ta_dir_tmp):
            for method in dirs:
                files_path = path_ta_dir_tmp + "/" + method
                for root, dirs, files in os.walk(files_path):
                    for file in files:
                        if file.endswith(".csv"):
                            arguments = re.split('[_.]', file)
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
                                        "TD4C - Cosine with Gradient"}})

    df.to_csv(output_file_name, index=False)


def merge_two_df(raw_df, ta_df, metrics):
    ta_df = ta_df.groupby(["classifier_name", "dataset_name", "method", "nb bins", "transformation_type"],
                          as_index=False).agg({k: np.mean for k in metrics})
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

    if max_val is not None:
        df = df.nlargest(max_val, "avg_ta")

    best_params = df[lst_group_by]
    keys = list(best_params.columns.values)
    i1 = ta_df.set_index(keys).index
    i2 = best_params.set_index(keys).index
    df = ta_df[i1.isin(i2)]

    return df


def create_fig(x, y, col, data, name, x_label, y_label='', legend='', hue=None, type="UCR", order=None, join=False,
               graph_num=1, colors=1):
    graph_aspect = 1.3

    if graph_num == 5:
        metrics = ['MCC', 'Cohen Kappa', 'F1 Score Macro', 'F1 Score Micro', 'F1 Score Weighted', 'Balanced Accuracy',
                   'AUC - ROC']
    elif graph_num == 7:
        metrics = ['Balanced Accuracy']
        data = data.loc[data['Evaluation Metric'] == "Balanced Accuracy"]
        graph_aspect = 3 if type == "UCR" else 2

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
            axis.set_xlabel(x_label, fontdict={'weight': 'bold'}, fontproperties=my_font_bold)
            axis.tick_params(labelleft=True, labelbottom=True)
            if legend != "":
                axis.legend(title=legend, prop={'family': 'Cambria'})

            axis.set_title(None)
            y_label_new = metrics[v] if y_label == '' else y_label

            plt.setp(axis.get_yticklabels(), fontproperties=my_font)
            plt.setp(axis.get_xticklabels(), rotation=80, fontproperties=my_font)

            axis.set_ylabel(y_label_new, fontproperties=my_font_bold)

            v += 1

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

    plt.savefig("test/" + type + "/" + name, bbox_inches='tight')


def melt_RawData_TA(df, lst, metrics):
    melt_ta = pd.melt(df, id_vars=lst, value_vars=metrics)

    df = df.drop(metrics, axis=1)
    df.columns = df.columns.str.replace('_raw_data', '')

    melt_raw = pd.melt(df, id_vars=lst, value_vars=metrics)

    # Merge the two df
    data = pd.concat([melt_raw.assign(Data='Raw Data'), melt_ta.assign(Data='TA')])

    return data


def metrics_best_combination_VS_raw_data(df, metrics, type="UCR"):
    data = melt_RawData_TA(df, ["nb bins", "method", "transformation_type"], metrics)

    data = data.rename(columns={"variable": "Evaluation Metric", "method": "Method",
                                "transformation_type": "Transformation Type"})

    create_fig(x="Evaluation Metric", y="value", col=None, data=data, name="TA vs Raw data", legend="TA vs Raw data",
               x_label='Method', y_label="Value", hue="Data", type=type, graph_num=5, colors=3)


def classifiers_best_combination_VS_raw_data(df, metrics, type="UCR"):
    data = melt_RawData_TA(df, ["nb bins", "method", "transformation_type", "classifier_name"], metrics)

    data = data.rename(columns={"variable": "Evaluation Metric", "classifier_name": "Classifier"})

    create_fig(x="Classifier", y="value", col="Evaluation Metric", data=data, name="TA vs Raw data - Classifier",
               x_label='Classifier', legend="TA vs Raw data", hue="Data", type=type, colors=3)


def datasets_best_combination_VS_raw_data(df, metrics, type="UCR"):
    data = melt_RawData_TA(df, ["nb bins", "method", "transformation_type", "dataset_name"], metrics)

    data = data.rename(columns={"variable": "Evaluation Metric", "dataset_name": "Dataset"})

    create_fig(x="Dataset", y="value", col="Evaluation Metric", data=data, name="TA vs Raw data - Datasets",
               x_label='Dataset', legend="TA vs Raw data", hue="Data", type=type, graph_num=7, colors=3)


def add_characteristic(characteristic_name, df):
    if characteristic_name == "lengths":
        df.loc[df.lengths < 81, 'groups_lengths'] = "< 81"
        df.loc[(df.lengths >= 81) & (df.lengths <= 250), 'groups_lengths'] = "81 - 250"
        df.loc[(df.lengths >= 251) & (df.lengths <= 450), 'groups_lengths'] = "251 - 450"
        df.loc[(df.lengths >= 451) & (df.lengths <= 700), 'groups_lengths'] = "451 - 700"
        df.loc[(df.lengths >= 701) & (df.lengths <= 1000), 'groups_lengths'] = "701 - 1000"
        df.loc[df.lengths > 1000, 'groups_lengths'] = " > 1000"

    else:
        df.loc[df.classes < 3, 'groups_classes'] = "Binary"
        df.loc[(df.classes >= 3) & (df.classes <= 10), 'groups_classes'] = "3 - 10"
        df.loc[(df.classes > 10), 'groups_classes'] = "> 10"

    return df


def dataset_characteristics(df, metrics, characteristics_df, characteristic_name, type, continuous=False):
    df = pd.merge(df, characteristics_df, on="dataset_name")

    fig_name = "TA vs Raw data - " + characteristic_name
    if continuous:
        df = df.rename(columns={characteristic_name: "groups_" + characteristic_name})
        fig_name += " - Continuous"
        order = None
        join = True
    else:
        df = add_characteristic(characteristic_name, df)
        join = None
        if characteristic_name == "classes":
            order = ["Binary", "3 - 10", "> 10"]
        else:
            order = ["< 81", "81 - 250", "251 - 450", "451 - 700", "701 - 1000", " > 1000"]

    data = melt_RawData_TA(df, ["nb bins", "method", "transformation_type", "groups_" + characteristic_name], metrics)

    data = data.rename(columns={"variable": "Evaluation Metric", "groups_" + characteristic_name: "Dataset " +
                                                                                                  characteristic_name})

    create_fig(x="Dataset " + characteristic_name, y="value", col="Evaluation Metric", data=data, name=fig_name,
               x_label="Dataset " + characteristic_name, legend="TA vs Raw data", hue="Data", type=type,
               order=order, join=join, colors=3)


def create_all_graphs(graph_numbers, create_csv=False, type="UCR"):
    config = ConfigClass()
    # Run this only once
    if create_csv:
        path_for_ta_dir = config.path + "ResultsProject/RawData/" + type + "/"
        concat_results(path_for_ta_dir, raw_data=True, type=type)

        path_for_ta_dir = config.path + "ResultsProject/AfterTA/" + type + "/"
        concat_results(path_for_ta_dir, type=type)

    # Open after TA data
    df_after_ta = pd.read_csv("results_of_all_combinations_" + type + "_new.csv", encoding="utf-8")
    # Open raw data
    raw_data_df = pd.read_csv("Raw_data_results_" + type + "_new.csv", encoding="utf-8")

    metrics = ['MCC', 'Cohen Kappa', 'F1 Score Macro', 'F1 Score Micro', 'F1 Score Weighted', 'Balanced Accuracy',
               'AUC - ROC']

    df = merge_two_df(raw_data_df, df_after_ta, metrics)

    for num in graph_numbers:
        # Graphs of the methods
        if num == 1:
            df = get_best_df_after_ta(df, metrics, ["nb bins", "method"], max_val=None)

            melt_df = pd.melt(df, id_vars=["nb bins", "method"], value_vars=metrics)
            melt_df = melt_df.rename(columns={"variable": "Evaluation Metric", "method": "Method"})
            create_fig(x="Method", y="value", col="Evaluation Metric", data=melt_df, name="Method",
                       x_label='Method', hue="nb bins", legend="Number of Symbols", type=type, colors=1)

        # Graphs of the top N methods
        elif num == 2:
            df = get_best_df_after_ta(df, metrics, ["nb bins", "method"], max_val=5)

            melt_df = pd.melt(df, id_vars=["nb bins", "method"], value_vars=metrics)
            melt_df = melt_df.rename(columns={"variable": "Evaluation Metric", "method": "Method"})

            create_fig(x="Method", y="value", col="Evaluation Metric", data=melt_df, name="Best 5 Methods",
                       x_label='Method', hue="nb bins", legend="Number of Symbols", type=type, colors=1)

        # Graph of the top transformation for the N best combination (nb bins & TA method)
        elif num == 3:
            df = get_best_df_after_ta(df, metrics, ["nb bins", "method"], max_val=5)

            melt_df = pd.melt(df, id_vars=["transformation_type"], value_vars=metrics)
            melt_df = melt_df.rename(columns={"variable": "Evaluation Metric",
                                              "transformation_type": "Transformation Type"})

            create_fig(x="Transformation Type", y="value", col="Evaluation Metric", data=melt_df,
                       name="Transformation for best combination of method & bins", x_label='Transformation Type',
                       legend="", type=type, order=["Discrete", "Symbol One-Hot", "Endpoint One-Hot"], colors=2)

        # Rank the transformation
        elif num == 4:
            melt_df = pd.melt(df, id_vars=["transformation_type"], value_vars=metrics)
            data = melt_df.rename(columns={"variable": "Evaluation Metric",
                                           "transformation_type": "Transformation Type"})

            create_fig(x="Transformation Type", y="value", col="Evaluation Metric", data=data,
                       name="Transformations", x_label="Transformation Type", legend="", type=type,
                       order=["Discrete", "Symbol One-Hot", "Endpoint One-Hot"], colors=2)

        # All the metric - Raw Data VS. best combination
        elif num == 5:
            df = get_best_df_after_ta(df, metrics, ["nb bins", "method", "transformation_type"], max_val=1)
            metrics_best_combination_VS_raw_data(df, metrics, type=type)

        # Classifiers - Raw Data VS. best combination
        elif num == 6:
            df = get_best_df_after_ta(df, metrics, ["nb bins", "method", "transformation_type"], max_val=1)
            classifiers_best_combination_VS_raw_data(df, metrics, type=type)

        # Datasets - Raw Data VS. best combination
        elif num == 7:
            df = get_best_df_after_ta(df, metrics, ["nb bins", "method", "transformation_type"], max_val=1)
            datasets_best_combination_VS_raw_data(df, metrics, type=type)

        elif num == 8:
            df = get_best_df_after_ta(df, metrics, ["nb bins", "method", "transformation_type"], max_val=1)

            length_dict = open_pickle(type + "_length")
            classes_dict = open_pickle(type + "_classes")

            # for length
            length_df = pd.DataFrame(list(length_dict.items()), columns=['dataset_name', 'lengths'])
            dataset_characteristics(df, metrics, length_df, "lengths", type=type, continuous=False)

            # for length - continuous
            length_df = pd.DataFrame(list(length_dict.items()), columns=['dataset_name', 'lengths'])
            dataset_characteristics(df, metrics, length_df, "lengths", type=type, continuous=True)

            # for classes
            classes_df = pd.DataFrame(list(classes_dict.items()), columns=['dataset_name', 'classes'])
            dataset_characteristics(df, metrics, classes_df, "classes", type=type, continuous=False)

            # for classes - continuous
            classes_df = pd.DataFrame(list(classes_dict.items()), columns=['dataset_name', 'classes'])
            dataset_characteristics(df, metrics, classes_df, "classes", type=type, continuous=True)


if __name__ == '__main__':
    create_all_graphs([1, 2, 3, 4, 5, 6, 7, 8], create_csv=False, type="MTS")
