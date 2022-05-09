import matplotlib

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

font_path = "/sise/robertmo-group/TA-DL-TSC/Cambria/CAMBRIA.TTC"
my_font = fm.FontProperties(fname=font_path)

font_path = "/sise/robertmo-group/TA-DL-TSC/Cambria/CAMBRIAB.TTF"
my_font_bold = fm.FontProperties(fname=font_path)

# Creates a unified file of all results for all combinations
def concat_results(path_ta_dir, raw_data=False):
    classifiers = ["cnn", "mlp", "mcdcnn", "fcn", "twiesn", "encoder", "inception", "lstm_fcn", "mlstm_fcn", "rocket"]

    columns_df = ['classifier_name', 'archive_name', 'dataset_name', 'Precision', 'Accuracy', 'Recall', 'MCC',
                  'Cohen Kappa', 'Learning Time', 'Predicting Time', 'F1 Score Macro', 'F1 Score Micro',
                  'F1 Score Weighted', 'Balanced Accuracy', 'AUC', "iteration"]

    output_file_name = "Raw_data_results.csv"
    if not raw_data:
        columns_df += ['method', "nb bins", "paa", "std", "max gap", "gradient_window_size"]
        output_file_name = 'results_of_all_combinations.csv'

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
                                method = method + " Gradient" if arguments[8] == "TRUE" else method
                                method = method + " Per Entity" if arguments[9] == "TRUE" else method
                                res_ta_data["method"] = method
                                res_ta_data["nb bins"] = arguments[2]
                                res_ta_data["paa"] = arguments[3]
                                res_ta_data["std"] = arguments[4]
                                res_ta_data["max gap"] = arguments[5]
                                res_ta_data["gradient_window_size"] = arguments[6]

                                if arguments[7] == "1":
                                    transformation_name = "Categorical"
                                elif arguments[7] == "2":
                                    transformation_name = "One-Hot"
                                else:
                                    transformation_name = "Endpoint"

                                res_ta_data["transformation_type"] = transformation_name

                    df = pd.concat([df, res_ta_data])

    df["Learning Time"] = df["Learning Time"] / 1000
    df["Predicting Time"] = df["Predicting Time"] / 1000

    # Rename raw data columns
    if raw_data:
        df.rename(columns=lambda x: x + "_raw_data", inplace=True)

    df.to_csv(output_file_name, index=False)


def get_best_df_after_ta(raw_data_df, df_after_ta, metrics, lst_group_by, max_val):
    # Mean val for group by
    dict_after_ta = {k: np.mean for k in metrics}
    dict_raw_data = {k + "_raw_data": np.mean for k in metrics}

    raw_data_df = raw_data_df.groupby(['classifier_name_raw_data', 'archive_name_raw_data'],
                                      as_index=False).agg(dict_raw_data)

    df_after_ta = df_after_ta.groupby(lst_group_by,
                                      as_index=False).agg(dict_after_ta)

    df_raw_mean = raw_data_df[list(dict_raw_data.keys())].mean()

    # Add the value of the raw data to all the rows of the after ta df
    for metric in list(dict_raw_data.keys()):
        df_after_ta[metric] = df_raw_mean[metric]

    merge_df = df_after_ta

    for v in metrics:
        merge_df["minus_" + v] = merge_df[v] - merge_df[v + "_raw_data"]

    merge_df["avg_raw"] = merge_df[["minus_" + i for i in metrics]].mean(axis=1)
    merge_df["avg_ta"] = merge_df[[i for i in metrics]].mean(axis=1)

    df = merge_df.groupby(lst_group_by, as_index=False).agg({**dict_after_ta, **dict_raw_data, **{"avg_raw": np.mean}})
    df_top_from_raw = df.nlargest(max_val, "avg_raw")

    df = merge_df.groupby(lst_group_by, as_index=False).agg({**dict_after_ta, **{"avg_ta": np.mean}})
    df_top_ta = df.nlargest(max_val, "avg_ta")

    return df_top_from_raw, df_top_ta


def create_fig_best_transformation(best_params, after_ta_df, metrics, graph_best_transformation= False ):
    dict_after_ta = {k: np.mean for k in metrics}
    if not graph_best_transformation:
        df =concat_after_df_with_best_df(after_ta_df, best_params, metrics, ["nb bins", "method", "transformation_type"])
    else:
        df = after_ta_df

    df = df.groupby(["transformation_type"], as_index=False).agg(dict_after_ta)

    melt_df_after = pd.melt(df, id_vars=["transformation_type"], value_vars=metrics)
    data = melt_df_after.rename(columns={"variable": "Evaluation Metric", "transformation_type": "Transformation Type"})

    create_fig(x="Transformation Type", y="value", col="Evaluation Metric", data=data, name="best_transformation",
               x_label="Transformation Type", legend="")


def create_fig_best_params_VS_raw_data(best_params, raw_data_df, metrics):
    select_list = [i + "_raw_data" for i in metrics]
    df_raw_mean = raw_data_df[select_list].mean()
    for metric in select_list:
        best_params[metric] = df_raw_mean[metric]

    melt_df_after = pd.melt(best_params, id_vars=["nb bins", "method", "transformation_type"], value_vars=metrics)

    best_params.drop(metrics, inplace=True, axis=1)
    best_params.columns = best_params.columns.str.replace('_raw_data', '')

    melt_before = pd.melt(best_params, id_vars=["nb bins", "method", "transformation_type"], value_vars=metrics)

    # Merge the two df
    data = pd.concat([melt_before.assign(Data='Raw Data'), melt_df_after.assign(Data='After TA')])

    data = data.rename(columns={"variable": "Evaluation Metric",
                                "method": "Method", "transformation_type": "Transformation Type"})

    create_fig(x="Evaluation Metric", y="value", col=None, data=data, name="TA vs Raw data",
               x_label='Method',y_label="Value", hue="Data")


def create_fig_best_params_VS_raw_data_classifier(best_params, raw_data_df,after_ta_df, metrics):
    df = concat_after_df_with_best_df(after_ta_df, best_params, metrics,
                                      ["nb bins", "method", "transformation_type", "classifier_name"])
    select_list = [i + "_raw_data" for i in metrics]
    df_raw_mean = raw_data_df[select_list].mean()
    for metric in select_list:
        df[metric] = df_raw_mean[metric]

    melt_df_after = pd.melt(df, id_vars=["nb bins", "method", "transformation_type", "classifier_name"], value_vars=metrics)

    df.drop(metrics, inplace=True, axis=1)
    df.columns = df.columns.str.replace('_raw_data', '')

    melt_before = pd.melt(df, id_vars=["nb bins", "method", "transformation_type", "classifier_name"], value_vars=metrics)

    # Merge the two df
    data = pd.concat([melt_before.assign(Data='Raw Data'), melt_df_after.assign(Data='After TA')])

    data = data.rename(columns={"variable": "Evaluation Metric", "classifier_name": "Classifier"})

    create_fig(x="Classifier", y="value", col="Evaluation Metric", data=data, name="TA vs Raw data - Classifier",
               x_label='Classifier',legend="TA vs Raw data",  hue="Data")


def create_fig_dataset_characteristics(best_params, raw_data_df , after_ta_df, metrics, characteristics_df, characteristic_name):
    df = concat_after_df_with_best_df(after_ta_df, best_params, metrics, ["nb bins", "method", "transformation_type", "dataset_name"])

    dict_raw_data = {k + "_raw_data": np.mean for k in metrics}

    raw_data_df = raw_data_df.groupby(['dataset_name_raw_data'],
                                      as_index=False).agg(dict_raw_data)

    raw_data_df = raw_data_df.rename(columns={"dataset_name_raw_data": "dataset_name"})
    df = pd.merge(raw_data_df, df, on=["dataset_name" ])

    df = pd.merge(df, characteristics_df, on="dataset_name")

    df = add_characteristic(characteristic_name, df)

    melt_df_after = pd.melt(df, id_vars=["nb bins", "method", "transformation_type", "groups_" + characteristic_name], value_vars=metrics)

    df.drop(metrics, inplace=True, axis=1)
    df.columns = df.columns.str.replace('_raw_data', '')

    melt_before = pd.melt(df, id_vars=["nb bins", "method", "transformation_type", "groups_" + characteristic_name], value_vars=metrics)

    # Merge the two df
    data = pd.concat([melt_before.assign(Data='Raw Data'), melt_df_after.assign(Data='After TA')])

    data = data.rename(columns={"variable": "Evaluation Metric", "groups_" + characteristic_name: "Dataset " + characteristic_name})

    create_fig(x="Dataset " + characteristic_name, y="value", col="Evaluation Metric", data=data,
               name="TA vs Raw data - " + characteristic_name,
               x_label="Dataset " + characteristic_name ,legend="TA vs Raw data",  hue="Data")


def concat_after_df_with_best_df(after_ta_df, best_params, metrics, group_by_lst):
    dict_after_ta = {k: np.mean for k in metrics}
    best_params = best_params.iloc[:, 0:2]
    after_ta_df = after_ta_df.groupby(group_by_lst, as_index=False).agg(
        dict_after_ta)
    keys = list(best_params.columns.values)
    i1 = after_ta_df.set_index(keys).index
    i2 = best_params.set_index(keys).index
    df = after_ta_df[i1.isin(i2)]
    return df


def add_characteristic(characteristic_name, df):
    if characteristic_name == "lengths":
        df.loc[df.lengths < 81, 'groups_lengths'] = "< 81"
        df.loc[(df.lengths >= 81) & (df.lengths <= 250), 'groups_lengths'] = "81 - 250"
        df.loc[(df.lengths >= 251) & (df.lengths <= 450), 'groups_lengths'] = "251 - 450"
        df.loc[(df.lengths >= 451) & (df.lengths <= 700), 'groups_lengths'] = "451 - 700"
        df.loc[(df.lengths >= 701) & (df.lengths <= 1000), 'groups_lengths'] = "701 - 1000"
        df.loc[df.lengths > 1000, 'groups_lengths'] = " > 1000"

    else:
        df.loc[df.classes < 3, 'groups_classes'] = "< 3"
        df.loc[(df.classes >= 3) & (df.classes <= 10), 'groups_classes'] = "3 - 10"
        df.loc[(df.classes > 10), 'groups_classes'] = "> 10"

    return df


def create_fig(x, y, col, data, name, x_label, y_label='' , legend='', hue=None):
    metrics = ['MCC', 'Cohen Kappa', 'F1 Score Macro', 'F1 Score Micro', 'F1 Score Weighted', 'Balanced Accuracy',
               'AUC']
    sns.set_palette(sns.color_palette(["#FF0B04", "#4374B3"]))
    g = sns.catplot(x=x, y=y,
                    col=col,
                    hue=hue,
                    data=data, kind="point",
                    sharey=False,
                    height=6, aspect=1.3, join=False)
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
            if y_label == '':
                y_label_new = metrics[v]
            else:
                y_label_new = y_label

            plt.setp(axis.get_yticklabels(), fontproperties=my_font)
            plt.setp(axis.get_xticklabels(), rotation=30, fontproperties=my_font)

            axis.set_ylabel(y_label_new, fontproperties=my_font_bold)

            # axis.set_ylim(0, 1)
            v += 1

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()
    plt.savefig(name, bbox_inches='tight')
    print()


def create_all_graphs(create_csv=False):
    config = ConfigClass()
    # Run this only once
    if create_csv:
        path_for_ta_dir = config.path + "ResultsProject/RawData/" + "UCR" + "/"
        concat_results(path_for_ta_dir, raw_data=True)

        path_for_ta_dir = config.path + "ResultsProject/AfterTA/" + "UCR" + "/"
        concat_results(path_for_ta_dir)

    # Open after TA data
    df_after_ta = pd.read_csv("Unified_results/results_of_all_combinations.csv", encoding="utf-8")
    # Open raw data
    raw_data_df = pd.read_csv("Unified_results/Raw_data_results.csv", encoding="utf-8")

    metrics = ['MCC', 'Cohen Kappa', 'F1 Score Macro', 'F1 Score Micro', 'F1 Score Weighted', 'Balanced Accuracy',
               'AUC']

    # --------------------------------Graph 1----------------------------------------------------------

    df_top_from_raw, df_top_ta= get_best_df_after_ta(raw_data_df,df_after_ta,metrics, ["nb bins", "method"], 3)
    melt_df_after = pd.melt(df_top_ta, id_vars= ["nb bins", "method"], value_vars=metrics)
    melt_df_after = melt_df_after.rename(columns={"variable": "Evaluation Metric", "method": "Method"})

    create_fig(x= "Method",y="value",col="Evaluation Metric",data=melt_df_after, name="Best_n_ta",
               x_label='Method',hue="nb bins",  legend="Number of Symbols")

    # -------------------------------------------------------------------------------------------------

    # --------------------------------Graph 2----------------------------------------------------------
    # df_top_from_raw, df_top_ta= get_best_df_after_ta(raw_data_df,df_after_ta,metrics, ["transformation_type"], 3)
    # melt_df_after = pd.melt(df_top_ta, id_vars= ["transformation_type"], value_vars=metrics)
    # melt_df_after = melt_df_after.rename(columns={"variable": "Evaluation Metric",
    #                                               "transformation_type": "Transformation Type"})
    #
    # create_fig(x= "Transformation Type",y="value",col="Evaluation Metric",data=melt_df_after, name="Best_transformation_with_best_ta",
    #            x_label='Transformation Type', legend="")
    # -------------------------------------------------------------------------------------------------

    # --------------------------------Graph 3----------------------------------------------------------
    # create_fig_best_transformation(None, df_after_ta, metrics, graph_best_transformation= True)
    # -------------------------------------------------------------------------------------------------

    # --------------------------------Graph 4----------------------------------------------------------
    # df_top_from_raw, df_top_ta = get_best_df_after_ta(raw_data_df, df_after_ta, metrics, ["nb bins", "method",
    #                                                                                       "transformation_type"], 1)
    # create_fig_best_params_VS_raw_data(df_top_ta, raw_data_df, metrics)
    # -------------------------------------------------------------------------------------------------

    # --------------------------------Graph 5----------------------------------------------------------
    # df_top_from_raw, df_top_ta = get_best_df_after_ta(raw_data_df, df_after_ta, metrics, ["nb bins", "method",
    #                                                                                       "transformation_type"], 1)
    # create_fig_best_params_VS_raw_data_classifier(df_top_ta,raw_data_df, df_after_ta, metrics )

    # -------------------------------------------------------------------------------------------------

    # --------------------------------Graph 6----------------------------------------------------------
    df_top_from_raw, df_top_ta = get_best_df_after_ta(raw_data_df, df_after_ta, metrics, ["nb bins", "method",
                                                                                          "transformation_type"], 1)
    # for length
    mts_length = open_pickle("mts_length")
    length_df = pd.DataFrame(list(mts_length.items()), columns=['dataset_name', 'lengths'])
    create_fig_dataset_characteristics(df_top_ta, raw_data_df, df_after_ta, metrics, length_df, "lengths")

    # for classes - todo need to check this
    mts_classes = open_pickle("mts_classes")
    classes_df = pd.DataFrame(list(mts_classes.items()), columns=['dataset_name', 'classes'])
    create_fig_dataset_characteristics(df_top_ta, raw_data_df, df_after_ta, metrics, classes_df, "classes")

    # -------------------------------------------------------------------------------------------------


create_all_graphs()
