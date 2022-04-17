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

# font_path = "C:\Windows\Fonts\cambria.ttc"
# my_font = fm.FontProperties(fname=font_path)

font_path = "C:\Windows\Fonts\cambria.ttc"
my_font = fm.FontProperties(fname=font_path)

#Creates a unified file of all results for all combinations
def concat_results(path_ta_dir, raw_data = False):
    classifiers = ["cnn", "mlp", "mcdcnn", "fcn", "twiesn", "encoder", "inception", "lstm_fcn"
                   ,"mlstm_fcn", "rocket", "tlenet"]

    columns_df = ['classifier_name', 'archive_name', 'dataset_name', 'precision', 'recall', 'accuracy',
     'mcc', "cohen_kappa", "f1_score_macro", "f1_score_micro",
     "f1_score_weighted", 'learning_time', 'predicting_time', "iteration"]

    output_file_name = "Raw_data_results.csv"
    if not raw_data:
        columns_df += ['method', "nb bins", "paa", "std", "max gap", "gradient_window_size"]
        output_file_name = 'results_of_all_combinations.csv'

    df = pd.DataFrame(columns=columns_df)
    for classifier in classifiers:
        path_ta_dir_tmp = path_ta_dir + classifier
        for root, dirs, files in os.walk(path_ta_dir_tmp):
            for method in dirs:
                files_path= path_ta_dir_tmp + "/" + method
                for root, dirs, files in os.walk(files_path):
                    for file in files:
                        if file.endswith(".csv"):
                            arguments = re.split('[_.]', file)
                            res_ta_data = pd.read_csv(root + "//" + file, sep=',', header=0, encoding="utf-8")
                            if not raw_data:
                                res_ta_data["method"] = method
                                res_ta_data["nb bins"] = arguments[2]
                                res_ta_data["paa"] = arguments[3]
                                res_ta_data["std"] = arguments[4]
                                res_ta_data["max gap"] = arguments[5]
                                res_ta_data["gradient_window_size"] = arguments[6]
                                res_ta_data["transformation_number"]= arguments[7]
                                res_ta_data["combination"] = arguments[8]

                    df = pd.concat([df, res_ta_data])

    df["learning_time"] = df["learning_time"] / 1000

    if raw_data:
        #todo !!
        df["learning_time"]=0
        df["predicting_time"] = 0

        df.rename(columns=lambda x: x + "_raw_data", inplace=True)

    #if not os.path.exists('Unified_results/' + output_file_name):
    df.to_csv(output_file_name, index=False)


def get_best_df_after_ta():
    # open after TA data
    df_after_ta = pd.read_csv("D:\\results_of_all_combinations_MTS.csv", encoding="utf-8")

    # For UCR ONLY
    # df_after_ta= df_after_ta.loc[(df_after_ta["transformation_number"] == 1)]

    # open raw data
    raw_data_df = pd.read_csv("D:\\Raw_data_results_MTS.csv", encoding="utf-8")
    # mean val for group by
    metrics = ['precision', 'recall', 'accuracy', 'mcc', "cohen_kappa", "f1_score_macro", "f1_score_micro",
                  "f1_score_weighted"]

    dict_after_ta = {k: np.mean for k in metrics}
    dict_raw_data = {k + "_raw_data": np.mean for k in metrics}

    z = {**dict_after_ta, **dict_raw_data}
    raw_data_df = raw_data_df.groupby(['classifier_name_raw_data', 'archive_name_raw_data'],
                                      as_index=False).agg(dict_raw_data)

    df_after_ta = df_after_ta.groupby(["nb bins", "combination", "method"],
                                      as_index=False).agg(dict_after_ta)

    metrics = ['precision', 'recall', 'accuracy', 'mcc', "cohen_kappa", "f1_score_macro", "f1_score_micro",
               "f1_score_weighted"]
    select_list = [i + "_raw_data" for i in metrics]
    df_raw_mean = raw_data_df[select_list].mean()

    for metric in select_list:
        df_after_ta[metric] = df_raw_mean[metric]

    merge_df = df_after_ta

    for v in metrics:
        merge_df["minus_" + v] = merge_df[v] - merge_df[v + "_raw_data"]

    merge_df["avg"] = merge_df[["minus_" + i for i in metrics]].mean(axis=1)
    merge_df_dict = {**dict_after_ta, **{"avg": np.mean}}

    df = merge_df.groupby(["nb bins", "method"], as_index=False).agg(merge_df_dict)
    df_top = df.nlargest(3, "avg")
    return df_top


def create_fig_best_params():
    best_params = get_best_df_after_ta()

    x = best_params.iloc[:,0:2].to_string(header=False,
                      index=False,
                      index_names=False).split('\n')
    vals = [' , '.join(ele.split()) for ele in x]

    raw_data_df = pd.read_csv("D:\\Raw_data_results_MTS.csv", encoding="utf-8")

    metrics = ['precision', 'recall', 'accuracy', 'mcc', "cohen_kappa", "f1_score_macro", "f1_score_micro",
               "f1_score_weighted"]
    select_list = [i + "_raw_data" for i in metrics]
    df_raw_mean = raw_data_df[select_list].mean()

    for metric in select_list:
        best_params[metric] = df_raw_mean[metric]

    # best_params = pd.DataFrame(best_params.iloc[0]).T

    colors = ["#FF0B04", "#4374B3"]

    concat_df = best_params.reset_index()

    melt_df_after = pd.melt(concat_df, id_vars=["index"], value_vars=metrics)

    concat_df.drop(metrics, inplace=True, axis=1)

    concat_df.columns = concat_df.columns.str.replace('_raw_data', '')

    melt_before = pd.melt(concat_df, id_vars=["index"], value_vars=metrics)

    # merge the two df
    data = pd.concat([melt_before.assign(Data='Raw Data'), melt_df_after.assign(Data='After TA')])
    data = data.rename(columns={"variable": "Evaluation Metric", "index": "Rating"})

    sns.set_palette(sns.color_palette(colors))
    g = sns.catplot(x="Evaluation Metric", y="value",
                    hue="Data", col="Rating",
                    data=data, kind="point",
                    sharey=False,
                    height=6, aspect=1.3, join=False)

    v = 0
    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            axis = g.axes[i, j]
            axis.set_xlabel('Evaluation Metric')
            #axis.set_title( str(v) + " - " + vals[v])
            axis.tick_params(labelleft=True, labelbottom=True)
            axis.set_xlabel("Evaluation Metric", fontdict={'weight': 'bold'})
            axis.set_ylim(0, 1)
            axis.set_ylabel(axis.get_ylabel(), fontdict={'weight': 'bold'})
            plt.setp(axis.get_xticklabels(), rotation=30)
            v += 1

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()
    plt.savefig("Rating", bbox_inches='tight')
    print()


def create_fig_best_transformation():
    best_params = get_best_df_after_ta()
    best_params = best_params.iloc[:, 0:2]

    raw_data_df = pd.read_csv("D:\\Raw_data_results_MTS.csv", encoding="utf-8")

    metrics = ['precision', 'recall', 'accuracy', 'mcc', "cohen_kappa", "f1_score_macro", "f1_score_micro",
               "f1_score_weighted"]
    select_list = [i + "_raw_data" for i in metrics]
    df_raw_mean = raw_data_df[select_list].mean()

    after_ta_df = pd.read_csv("D:\\results_of_all_combinations_MTS.csv", encoding="utf-8")

    dict_after_ta = {k: np.mean for k in metrics}
    after_ta_df = after_ta_df.groupby(["nb bins", "method", "transformation_number"], as_index=False).agg(dict_after_ta)

    keys = list(best_params.columns.values)
    i1 = after_ta_df.set_index(keys).index
    i2 = best_params.set_index(keys).index
    df = after_ta_df[i1.isin(i2)]

    df = df.groupby(["transformation_number"], as_index=False).agg(dict_after_ta)

    for metric in select_list:
        df[metric] = df_raw_mean[metric]

    colors = ["#FF0B04", "#4374B3"]

    concat_df = df

    melt_df_after = pd.melt(concat_df, id_vars=["transformation_number"], value_vars=metrics)

    concat_df.drop(metrics, inplace=True, axis=1)

    concat_df.columns = concat_df.columns.str.replace('_raw_data', '')

    melt_before = pd.melt(concat_df, id_vars=["transformation_number"], value_vars=metrics)

    # merge the two df
    data = pd.concat([melt_before.assign(Data='Raw Data'), melt_df_after.assign(Data='After TA')])
    data = data.rename(columns={"variable": "Evaluation Metric", "transformation_number": "Transformation Number"})

    sns.set_palette(sns.color_palette(colors))
    g = sns.catplot(x="Evaluation Metric", y="value",
                    hue="Data", col="Transformation Number",
                    data=data, kind="point",
                    sharey=False,
                    height=6, aspect=1.3, join=False)

    v = 0
    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            axis = g.axes[i, j]
            axis.set_xlabel('Evaluation Metric')
            #axis.set_title( str(v) + " - " + vals[v])
            axis.tick_params(labelleft=True, labelbottom=True)
            axis.set_xlabel("Evaluation Metric", fontdict={'weight': 'bold'})
            axis.set_ylim(0, 1)
            axis.set_ylabel(axis.get_ylabel(), fontdict={'weight': 'bold'})
            plt.setp(axis.get_xticklabels(), rotation=30)
            v += 1

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()
    plt.savefig("Rating_Transformation", bbox_inches='tight')
    print()


def find_best_params():
    # open after TA data
    df_after_ta = pd.read_csv("C:\\results_of_all_combinations.csv", encoding="utf-8")
    df_after_ta = df_after_ta.loc[(df_after_ta["transformation_number"]==1)]

    # open raw data
    raw_data_df = pd.read_csv("Unified_results/Raw_data_results.csv", encoding="utf-8")
    # mean val for group by
    metrics = ['precision', 'recall', 'accuracy',
                  'mcc', "cohen_kappa", "f1_score_macro", "f1_score_micro",
                  "f1_score_weighted"]

    dict_after_ta= {k:np.mean  for k in metrics}
    dict_raw_data = {k+"_raw_data": np.mean for k in metrics}

    z = {**dict_after_ta, **dict_raw_data}
    raw_data_df = raw_data_df.groupby(['classifier_name_raw_data', 'archive_name_raw_data'], as_index=False).agg(dict_raw_data)

    df_after_ta = df_after_ta.groupby(['classifier_name', "nb bins", "transformation_number",
                     "combination", "method" ], as_index=False).agg(dict_after_ta)

    #merge the after ta and the raw data
    merge_df = pd.merge(raw_data_df, df_after_ta, left_on=["classifier_name_raw_data"],
                        right_on = ["classifier_name"], how='left')

    merge_df = merge_df[merge_df['classifier_name'].notna()]

    for v in metrics:
        merge_df["minus_" + v ] = merge_df[v] - merge_df[ v + "_raw_data"]

    merge_df["avg"] = merge_df[["minus_"+ i for i in metrics]].mean(axis=1)
    merge_df_dict={**dict_after_ta, **{"avg": np.mean}}

    # filter all the positive rows
    df = merge_df.groupby(["nb bins", "method"], as_index=False).agg(merge_df_dict)

    #get best parms
    #df_max = df.iloc[df["avg"].idxmax()]
    #get top n parms
    df_top = df.nlargest(3,"avg")

    # -------for title -----
    x = df_top.iloc[:,0:2].to_string(header=False,
                      index=False,
                      index_names=False).split('\n')
    vals = [' , '.join(ele.split()) for ele in x]
    # ------------------

    colors = ["#FF0B04", "#4374B3"]
    concat_df = pd.concat([df_top, merge_df], axis=1, join="inner")
    select_list=[ i +"_raw_data" for i in metrics] + metrics
    select_col = concat_df[select_list]
    select_col["method"]=df_top["method"]
    select_col["nb bins"] = df_top["nb bins"]
    concat_df= select_col.reset_index()

    melt_df_after = pd.melt(concat_df, id_vars=["index"], value_vars=metrics)

    concat_df.drop(metrics, inplace=True, axis=1)

    concat_df.columns = concat_df.columns.str.replace('_raw_data', '')

    melt_before = pd.melt(concat_df, id_vars=["index"],
                          value_vars = metrics)
    # merge the two df
    data = pd.concat([melt_before.assign(Data='Raw Data'), melt_df_after.assign(Data='After TA')])
    data = data.rename(columns={"variable": "Evaluation Metric", "index": "Rating"})

    sns.set_palette(sns.color_palette(colors))
    g = sns.catplot(x="Evaluation Metric", y="value",
                    hue="Data", col="Rating",
                    data=data, kind="point",
                    sharey=False,
                    height=6, aspect=1.3, join=False)
    v=0
    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            axis = g.axes[i, j]
            axis.set_xlabel('Evaluation Metric')
            axis.set_title( str(v) + " - " + vals[v])
            axis.tick_params(labelleft=True, labelbottom=True)
            axis.set_xlabel("Evaluation Metric", fontdict={'weight': 'bold'})
            axis.set_ylabel(axis.get_ylabel(), fontdict={'weight': 'bold'})
            plt.setp(axis.get_xticklabels(), rotation=30)
            v+=1

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig("Rating", bbox_inches='tight')
    print()


def create_all_graphs(row_data= False):
    config = ConfigClass()
    # run only once
    if row_data:
        path_for_ta_dir = config.path + "ResultsProject/RawData/" + "UCR" + "/"
        concat_results(path_for_ta_dir, raw_data=row_data)
    else:
        path_for_ta_dir = config.path +"ResultsProject/AfterTA/"+ "UCR" +"/"
        concat_results(path_for_ta_dir)


#create_all_graphs(row_data=True)
#create_all_graphs(row_data=False)
#graphs_for_all_datasets()
# create_fig_best_params()
create_fig_best_transformation()
