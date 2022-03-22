import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


def open_pickle(name):
    file = open( name + ".pkl", "rb")
    data = pickle.load(file)
    return data


def concat_results(path_ta_dir):
    classifiers = ["cnn", "mlp", "mcdcnn", "fcn", "twiesn"]
    columns_df = ['classifier_name', 'archive_name', 'dataset_name', 'precision', 'accuracy', 'recall', 'duration',
                  "iteration", 'method', "nb bins", "paa", "std", "max gap", "gradient_window_size"]

    df = pd.DataFrame(columns=columns_df)
    for classifier in classifiers:
        path_ta_dir_tmp= path_ta_dir+ classifier
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
    # calculet f score
    df["f_score"] = ((df["recall"] * df["precision"]) /
                                (df["recall"] + df["precision"])) * 2

    if os.path.exists('results_new.csv'):
        df.to_csv("results_new.csv", index=False, mode='a', header=0)
    else:
        df.to_csv('results_new.csv', index=False)


def graph_1(df_after_ta, path_file_raw_data):

    df_after_ta = pd.read_csv(df_after_ta, encoding="utf-8")
    df_after_ta = df_after_ta.loc[(df_after_ta.paa == 1)]
    df_after_ta = df_after_ta.rename(columns={"f_score": "F_Score", "accuracy": "Accuracy",
                                    "duration": "Learning Time (ms)", "max gap": "Interpolation Gap"})
    # open raw data
    raw_data_df = pd.read_csv(path_file_raw_data, encoding="utf-8")

    raw_data_df["duration"] = raw_data_df["duration"] / 1000
    # calculet f score
    raw_data_df["f_score_b"] = ((raw_data_df["recall"] * raw_data_df["precision"]) /
                                (raw_data_df["recall"] + raw_data_df["precision"])) * 2

    merge_df = pd.merge(raw_data_df, df_after_ta, on=["dataset_name", "classifier_name", "archive_name", "iteration"])

    # rename columns
    merge_df = merge_df.rename(columns={"accuracy": "accuracy_b", "max gap": "max gap_b","duration":"duration_b" })


    melt_df_after = pd.melt(merge_df, id_vars=["dataset_name", "classifier_name"],
                            value_vars=['F_Score', "Accuracy", "Learning Time (ms)"])

    merge_df.drop(['F_Score', "Accuracy", "Learning Time (ms)","Interpolation Gap" ], inplace=True, axis=1)

    merge_df = merge_df.rename(columns={"f_score_b": "F_Score", "accuracy_b": "Accuracy",
                                    "duration_b": "Learning Time (ms)","max gap_b":"Interpolation Gap"  })

    melt_before = pd.melt(merge_df, id_vars=["dataset_name", "classifier_name"],
                          value_vars=['F_Score', "Accuracy", "Learning Time (ms)"])

    data = pd.concat([melt_before.assign(Data='Raw Data'),
                      melt_df_after.assign(Data='After TA')])

    sns.set_theme(style="darkgrid")
    colors = ["#FF0B04", "#4374B3"]
    sns.set_palette(sns.color_palette(colors))

    g = sns.FacetGrid(data,row="dataset_name", col="variable", hue="Data", height=6, aspect=1.8)
    g = g.map(sns.pointplot, "classifier_name", "value", ci=95, err_style="bars", s=40)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    for axis in g.axes.flat:
        axis.tick_params(labelleft=True, labelbottom=True)

    plt.subplots_adjust(wspace=0.3, hspace=0.15)
    plt.show()
    #
    # plt.savefig("plot/" + "top.png", bbox_inches='tight')
    # plt.clf()

    print()



def graph_2(df_after_ta):
    df_after_ta = pd.read_csv(df_after_ta, encoding="utf-8")
    # df_after_ta = df_after_ta.loc[(df_after_ta.paa == 1)]
    df_after_ta = df_after_ta.rename(columns={"f_score": "F_Score", "accuracy": "Accuracy",
                                    "duration": "Learning Time (ms)", "max gap": "Interpolation Gap"})

    columns = ['classifier_name', 'archive_name', 'dataset_name', 'method', "nb bins", "paa", "std",
                  "Interpolation Gap", "gradient_window_size"]
    df_after_ta = df_after_ta.groupby(columns, as_index=False).agg \
        ({"F_Score": np.mean, "Accuracy": np.mean, "Learning Time (ms)": np.mean})

    melt_df_after = pd.melt(df_after_ta, id_vars=['method', "nb bins"],
                            value_vars=['F_Score', "Accuracy", "Learning Time (ms)"])

    sns.set_theme(style="darkgrid")
    colors = ["#FF0B04", "#4374B3"]
    sns.set_palette(sns.color_palette(colors))

    g = sns.FacetGrid(melt_df_after, col="variable", hue="nb bins", height=6, aspect=1.3)
    g = g.map(sns.scatterplot, "method", "value", ci=95)

    #g.add_legend(title="Number of bins")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)

    for axis in g.axes.flat:
        axis.tick_params(labelleft=True, labelbottom=True)

    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    plt.show()
    #
    # plt.savefig("plot/" + "top.png", bbox_inches='tight')
    # plt.clf()

    print()


def create_all_graphs():
    path = "C:\\Users\\Shaha\\Desktop\\ResultsAfterTA\\"
    # concat_results(path)
    graph_1("results_new.csv", "C:\\Users\\Shaha\\Desktop\\raw_data_results.csv")


# create_all_graphs()

file = open("C:\\Users\\Shaha\\Desktop\\running_dict.pkl", "rb")
data = pickle.load(file)
print()

