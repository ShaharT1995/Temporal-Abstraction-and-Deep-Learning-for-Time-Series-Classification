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


def get_best_df_after_ta(raw_data_df, df_after_ta, metrics, lst_group_by, max_val=None):
    # Mean val for group by
    # dict_after_ta = {k: [np.mean, 'std'] for k in metrics}
    dict_after_ta = {k: np.mean for k in metrics}
    dict_raw_data = {k + "_raw_data": np.mean for k in metrics}

    raw_data_df = raw_data_df.groupby(['classifier_name_raw_data', 'archive_name_raw_data'],
                                      as_index=False).agg(dict_raw_data)

    df_after_ta = df_after_ta.groupby(lst_group_by, as_index=False).agg(dict_after_ta)

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
    if max_val is not None:
        df_top_from_raw = df.nlargest(max_val, "avg_raw")
    else:
        df_top_from_raw = df

    df = merge_df.groupby(lst_group_by, as_index=False).agg({**dict_after_ta, **{"avg_ta": np.mean}})
    if max_val is not None:
        df_top_ta = df.nlargest(max_val, "avg_ta")
    else:
        df_top_ta = df

    return df_top_from_raw, df_top_ta


def create_fig_best_transformation(best_params, after_ta_df, metrics, graph_best_transformation=False, type="UCR"):
    dict_after_ta = {k: np.mean for k in metrics}
    if not graph_best_transformation:
        df = concat_after_df_with_best_df(after_ta_df, best_params, metrics,
                                          ["nb bins", "method", "transformation_type"], number_of_cols=2)
    else:
        df = after_ta_df

    df = df.groupby(["transformation_type"], as_index=False).agg(dict_after_ta)

    melt_df_after = pd.melt(df, id_vars=["transformation_type"], value_vars=metrics)
    data = melt_df_after.rename(columns={"variable": "Evaluation Metric", "transformation_type": "Transformation Type"})

    create_fig(x="Transformation Type", y="value", col="Evaluation Metric", data=data, name="best_transformation",
               x_label="Transformation Type", legend="", type=type, order=["Discrete", "Symbol One-Hot",
                                                                           "Endpoint One-Hot"])


def create_fig_best_params_VS_raw_data(best_params, raw_data_df, metrics, type="UCR"):
    select_list = [i + "_raw_data" for i in metrics]

    df_raw_mean = raw_data_df[select_list].mean()
    for metric in select_list:
        best_params[metric] = df_raw_mean[metric]

    melt_df_after = pd.melt(best_params, id_vars=["nb bins", "method", "transformation_type"], value_vars=metrics)

    best_params.drop(metrics, inplace=True, axis=1)
    best_params.columns = best_params.columns.str.replace('_raw_data', '')

    melt_before = pd.melt(best_params, id_vars=["nb bins", "method", "transformation_type"], value_vars=metrics)

    # Merge the two df
    data = pd.concat([melt_before.assign(Data='Raw Data'), melt_df_after.assign(Data='TA')])

    data = data.rename(columns={"variable": "Evaluation Metric",
                                "method": "Method", "transformation_type": "Transformation Type"})

    create_fig(x="Evaluation Metric", y="value", col=None, data=data, name="TA vs Raw data",
               x_label='Method', y_label="Value", hue="Data", type=type)


def create_fig_best_params_VS_raw_data_classifier(best_params, raw_data_df, after_ta_df, metrics, type="UCR"):
    df = concat_after_df_with_best_df(after_ta_df, best_params, metrics,
                                      ["nb bins", "method", "transformation_type", "classifier_name"], number_of_cols=3)

    dict_raw_data = {k + "_raw_data": np.mean for k in metrics}
    raw_data_df = raw_data_df.groupby(['classifier_name_raw_data'], as_index=False).agg(dict_raw_data)

    raw_data_df = raw_data_df.rename(columns={"classifier_name_raw_data": "classifier_name"})
    df = pd.merge(df, raw_data_df, on='classifier_name')

    melt_df_after = pd.melt(df, id_vars=["nb bins", "method", "transformation_type", "classifier_name"],
                            value_vars=metrics)

    df.drop(metrics, inplace=True, axis=1)
    df.columns = df.columns.str.replace('_raw_data', '')

    melt_before = pd.melt(df, id_vars=["nb bins", "method", "transformation_type", "classifier_name"],
                          value_vars=metrics)

    # Merge the two df
    data = pd.concat([melt_before.assign(Data='Raw Data'), melt_df_after.assign(Data='TA')])

    data = data.rename(columns={"variable": "Evaluation Metric", "classifier_name": "Classifier"})

    create_fig(x="Classifier", y="value", col="Evaluation Metric", data=data, name="TA vs Raw data - Classifier",
               x_label='Classifier', legend="TA vs Raw data", hue="Data", type=type)



def create_fig_best_params_VS_raw_data_datasets(best_params, raw_data_df, after_ta_df, metrics, type="UCR"):

    df = concat_after_df_with_best_df(after_ta_df, best_params, metrics,
                                      ["nb bins", "method", "transformation_type", "dataset_name"], number_of_cols=3)

    dict_raw_data = {k + "_raw_data": np.mean for k in metrics}
    raw_data_df = raw_data_df.groupby(['dataset_name_raw_data'], as_index=False).agg(dict_raw_data)

    raw_data_df = raw_data_df.rename(columns={"dataset_name_raw_data": "dataset_name"})
    df = pd.merge(df, raw_data_df, on='dataset_name')

    melt_df_after = pd.melt(df, id_vars=["nb bins", "method", "transformation_type", "dataset_name"],
                            value_vars=metrics)

    df.drop(metrics, inplace=True, axis=1)
    df.columns = df.columns.str.replace('_raw_data', '')

    melt_before = pd.melt(df, id_vars=["nb bins", "method", "transformation_type", "dataset_name"],
                          value_vars=metrics)

    # Merge the two df
    data = pd.concat([melt_before.assign(Data='Raw Data'), melt_df_after.assign(Data='TA')])

    data = data.rename(columns={"variable": "Evaluation Metric", "dataset_name": "Dataset"})

    create_fig(x="Dataset", y="value", col="Evaluation Metric", data=data, name="TA vs Raw data - Dataset",
               x_label='Dataset', legend="TA vs Raw data", hue="Data", type=type)


def create_fig_dataset_characteristics(best_params, raw_data_df, after_ta_df, metrics, characteristics_df,
                                       characteristic_name, type="UCR", continuous=False):
    df = concat_after_df_with_best_df(after_ta_df, best_params, metrics,
                                      ["nb bins", "method", "transformation_type", "dataset_name"], number_of_cols=3)

    dict_raw_data = {k + "_raw_data": np.mean for k in metrics}

    raw_data_df = raw_data_df.groupby(['dataset_name_raw_data'],
                                      as_index=False).agg(dict_raw_data)

    raw_data_df = raw_data_df.rename(columns={"dataset_name_raw_data": "dataset_name"})
    df = pd.merge(raw_data_df, df, on=["dataset_name"])
    df = pd.merge(df, characteristics_df, on="dataset_name")

    if continuous:
        df = df.rename(columns={characteristic_name: "groups_" + characteristic_name})
        fig_name= "TA vs Raw data - " + characteristic_name + " - Continuous"
        order = None
        join= True
    else:
        df = add_characteristic(characteristic_name, df)
        fig_name= "TA vs Raw data - " + characteristic_name
        join = None
        if characteristic_name == "classes":
            order = ["Binary", "3 - 10", "> 10"]
        else:
            order = ["< 81", "81 - 250", "251 - 450", "451 - 700", "701 - 1000", " > 1000"]

    melt_df_after = pd.melt(df, id_vars=["nb bins", "method", "transformation_type", "groups_" + characteristic_name],
                            value_vars=metrics)

    df.drop(metrics, inplace=True, axis=1)
    df.columns = df.columns.str.replace('_raw_data', '')

    melt_before = pd.melt(df, id_vars=["nb bins", "method", "transformation_type", "groups_" + characteristic_name],
                          value_vars=metrics)

    # Merge the two df
    data = pd.concat([melt_before.assign(Data='Raw Data'), melt_df_after.assign(Data='TA')])

    data = data.rename(
        columns={"variable": "Evaluation Metric", "groups_" + characteristic_name: "Dataset " + characteristic_name})


    create_fig(x="Dataset " + characteristic_name, y="value", col="Evaluation Metric", data=data,
               name=fig_name,
               x_label="Dataset " + characteristic_name, legend="TA vs Raw data", hue="Data", type=type,
               order=order, join=join )



def concat_after_df_with_best_df(after_ta_df, best_params, metrics, group_by_lst, number_of_cols):
    dict_after_ta = {k: np.mean for k in metrics}
    best_params = best_params.iloc[:, 0:number_of_cols]
    after_ta_df = after_ta_df.groupby(group_by_lst, as_index=False).agg(dict_after_ta)
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
        df.loc[df.classes < 3, 'groups_classes'] = "Binary"
        df.loc[(df.classes >= 3) & (df.classes <= 10), 'groups_classes'] = "3 - 10"
        df.loc[(df.classes > 10), 'groups_classes'] = "> 10"

    return df


def create_fig(x, y, col, data, name, x_label, y_label='', legend='', hue=None, type="UCR", order=None, join=False):
    # metrics = ['MCC', 'Cohen Kappa', 'F1 Score Macro', 'F1 Score Micro', 'F1 Score Weighted', 'Balanced Accuracy',
    #            'AUC - ROC']

    metrics = ['Balanced Accuracy', "AUC - ROC"]
    data = data.loc[(data['Evaluation Metric'] == "AUC - ROC") | (data['Evaluation Metric'] == "Balanced Accuracy")]
    # data = data.loc[(data['Evaluation Metric'] == "Balanced Accuracy")]
    # sns.set_palette(sns.color_palette(["#FF0B04", "#4374B3"]))
    # sns.set_palette(sns.color_palette(["#CD37CB", "#26AA1B"]))
    sns.set_palette(sns.color_palette(["#D422AE", "#D422AE"]))
    # CD37CB
    # 26AA1B
    g = sns.catplot(x=x, y=y,
                    col=col,
                    hue=hue,
                    data=data,
                    kind="point",
                    sharey=False,
                    height=6,
                    aspect=1.3,
                    join=join,
                    order=order,
                    legend=False,
                    scale=1.3
                    )
    v = 0

    # plt.rcParams["font.family"] = "Cambria"
    # plt.legend(title="Data", prop={'family': 'Cambria'})

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
            plt.setp(axis.get_xticklabels(), rotation=80, fontproperties=my_font)

            axis.set_ylabel(y_label_new, fontproperties=my_font_bold)

            # axis.set_ylim(0, 1)
            v += 1

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

    plt.savefig(type + "/" + name, bbox_inches='tight')


def create_all_graphs(graph_num, create_csv=False, type="UCR"):
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
    # --------------------------------Graph 1----------------------------------------------------------
    if graph_num == 1:
        df_top_from_raw, df_top_ta = get_best_df_after_ta(raw_data_df, df_after_ta, metrics, ["nb bins", "method"])
        melt_df_after = pd.melt(df_top_ta, id_vars=["nb bins", "method"], value_vars=metrics)
        melt_df_after = melt_df_after.rename(columns={"variable": "Evaluation Metric", "method": "Method"})

        create_fig(x="Method", y="value", col="Evaluation Metric", data=melt_df_after, name="Method",
                   x_label='Method', hue="nb bins", legend="Number of Symbols", type=type)
    # --------------------------------Graph 2----------------------------------------------------------
    elif graph_num == 2:
        df_top_from_raw, df_top_ta = get_best_df_after_ta(raw_data_df, df_after_ta,metrics, ["nb bins", "method"], 5)
        melt_df_after = pd.melt(df_top_ta, id_vars=["nb bins", "method"], value_vars=metrics)
        melt_df_after = melt_df_after.rename(columns={"variable": "Evaluation Metric", "method": "Method"})

        create_fig(x="Method", y="value", col="Evaluation Metric", data=melt_df_after, name="Best_n_ta",
                   x_label='Method', hue="nb bins", legend="Number of Symbols", type=type)

    # --------------------------------Graph 3----------------------------------------------------------
    elif graph_num == 3:
        df_top_from_raw, df_top_ta = get_best_df_after_ta(raw_data_df, df_after_ta, metrics, ["transformation_type"], 5)
        melt_df_after = pd.melt(df_top_ta, id_vars=["transformation_type"], value_vars=metrics)
        melt_df_after = melt_df_after.rename(columns={"variable": "Evaluation Metric",
                                                      "transformation_type": "Transformation Type"})

        create_fig(x="Transformation Type", y="value", col="Evaluation Metric", data=melt_df_after,
                   name="Best_transformation_with_best_ta", x_label='Transformation Type', legend="", type=type,
                   order=["Categorical", "One-Hot", "Endpoint"])

    # # --------------------------------Graph 4----------------------------------------------------------
    elif graph_num == 4:
        create_fig_best_transformation(None, df_after_ta, metrics, graph_best_transformation=True, type=type)

    # --------------------------------Graph 5----------------------------------------------------------
    elif graph_num == 5:
        df_top_from_raw, df_top_ta = get_best_df_after_ta(raw_data_df, df_after_ta, metrics, ["nb bins", "method",
                                                                                              "transformation_type"], 1)
        create_fig_best_params_VS_raw_data(df_top_ta, raw_data_df, metrics, type=type)

    # --------------------------------Graph 6----------------------------------------------------------
    elif graph_num == 6:
        df_top_from_raw, df_top_ta = get_best_df_after_ta(raw_data_df, df_after_ta, metrics, ["nb bins", "method",
                                                                                              "transformation_type"], 1)
        create_fig_best_params_VS_raw_data_classifier(df_top_ta, raw_data_df, df_after_ta, metrics, type=type)
        create_fig_best_params_VS_raw_data_datasets(df_top_ta, raw_data_df, df_after_ta, metrics, type=type)

    # # --------------------------------Graph 6----------------------------------------------------------
    elif graph_num == 7:
        df_top_from_raw, df_top_ta = get_best_df_after_ta(raw_data_df, df_after_ta, metrics, ["nb bins", "method",
                                                                                            "transformation_type"], 1)
        # for length
        length_dict = open_pickle(type + "_length")
        length_df = pd.DataFrame(list(length_dict.items()), columns=['dataset_name', 'lengths'])
        create_fig_dataset_characteristics(df_top_ta, raw_data_df, df_after_ta, metrics, length_df, "lengths", type=type)

        # for length - continuous
        length_dict = open_pickle(type + "_length")
        length_df = pd.DataFrame(list(length_dict.items()), columns=['dataset_name', 'lengths'])
        create_fig_dataset_characteristics(df_top_ta, raw_data_df, df_after_ta, metrics, length_df, "lengths",
                                                      type=type, continuous=True)

        # for classes
        classes_dict = open_pickle(type + "_classes")
        classes_df = pd.DataFrame(list(classes_dict.items()), columns=['dataset_name', 'classes'])
        create_fig_dataset_characteristics(df_top_ta, raw_data_df, df_after_ta, metrics, classes_df, "classes", type=type)

        # for classes - continuous
        classes_dict = open_pickle(type + "_classes")
        classes_df = pd.DataFrame(list(classes_dict.items()), columns=['dataset_name', 'classes'])
        create_fig_dataset_characteristics(df_top_ta, raw_data_df, df_after_ta, metrics, classes_df, "classes",
                                                         type=type,  continuous=True)


if __name__ == '__main__':
    create_all_graphs(6, create_csv=True, type="UCR")
