import pickle
import re
from builtins import print
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sns

def open_pickle(name):
    file = open(name + ".pkl", "rb")
    data = pickle.load(file)
    return data

def concat_results(path_ta_dir):
    columns_df = ['classifier_name', 'archive_name', 'dataset_name',
               'precision', 'accuracy', 'recall', 'duration',"iteration", 'method', "nb bins", "paa", "std",
                               "max gap", "gradient_window_size"]

    df = pd.DataFrame(columns=columns_df)
    for root, dirs, files in os.walk(path_ta_dir):
        for file in files:
            if file.endswith(".csv"):
                arguments = re.split('[_.]', file)
                res_ta_data = pd.read_csv(root + "//" + file, sep=',', header=0, encoding="utf-8")
                res_ta_data["method"]= arguments[1]
                res_ta_data["nb bins"]= arguments[2]
                res_ta_data["paa"]=arguments[3]
                res_ta_data["std"]= arguments[4]
                res_ta_data["max gap"]= arguments[5]
                res_ta_data["gradient_window_size"]= arguments[6]
                df= pd.concat([df,res_ta_data])

    if os.path.exists( 'results_new.csv'):
        df.to_csv( "results_new.csv", index=False, mode='a', header=0)
    else:
        df.to_csv('results_new.csv', index=False)


def merge_accuracy(x, field="duration"):
    before = field+"_before"
    merged = str(round(x[field], 2)) + " / " + str(round(x[before], 2)) + " (" + str(round(
        x[field] - x[before], 2)) + ")"
    return merged


def set_df_for_graphs_and_tables(path_file_raw_data, path_after_ta_df, measures,method):
    #open raw data
    raw_data_df = pd.read_csv(path_file_raw_data, encoding="utf-8")
    #rename columns
    raw_data_df = raw_data_df.rename(columns={measures: measures+ "_before"})
    raw_data_df = raw_data_df.rename(columns={"duration": "duration_before"})
    raw_data_df["duration_before"]=raw_data_df["duration_before"]/1000
    raw_data_df = raw_data_df.rename(columns={"max gap":"Interpolation gap"})
    #calculet f score
    raw_data_df["f_score"]=((raw_data_df["recall"]*raw_data_df["precision"])/(raw_data_df["recall"]+raw_data_df["precision"]))*2

    #open data after ta
    data_after_ta_df = pd.read_csv(path_after_ta_df, encoding="utf-8")
    #remane columns
    data_after_ta_df = data_after_ta_df.rename(columns={"max gap": "Interpolation gap"})
    #calculet f score
    data_after_ta_df["f_score_ta"]=((data_after_ta_df["recall"]*data_after_ta_df["precision"])/(data_after_ta_df["recall"]+data_after_ta_df["precision"]))*2
    data_after_ta_df["duration"] = data_after_ta_df["duration"] / 1000
    df = pd.merge(raw_data_df, data_after_ta_df, on=["dataset_name", "classifier_name", "archive_name", "iteration"])
    df = df.loc[df.method == method]

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

    res_df = merged.groupby(["classifier_name", "archive_name","groups_lengths","paa","nb bins","Interpolation gap","method"], as_index=False).agg \
        ({measures: np.mean, measures+ "_before": np.mean})
    return res_df, merged


def create_graphs(res_df,merge, param, measures = "accuracy", m=""):
    ucr_df = res_df.loc[res_df.archive_name == "UCRArchive_2018"].iloc[:: -1]
    mts_df = res_df.loc[res_df.archive_name == "mts_archive"].iloc[:: -1]
    merge = merge.groupby(["classifier_name", "archive_name", "groups_lengths","paa"], as_index=False).mean()

    melt_df = pd.melt(merge, id_vars=['groups_lengths',"paa"], value_vars=['f_score_ta',"accuracy","duration"])

    melt_before = pd.melt(merge, id_vars=['groups_lengths',"paa"], value_vars=['f_score',"accuracy_before","duration_before"])

    data = pd.concat([melt_df.assign(frame='df1'),
                      melt_before.assign(frame='df2')])

    for classifier in ["mlp"]:#todo
        ucr_classifier = ucr_df.loc[ucr_df.classifier_name == classifier]
        mts_classifier = mts_df.loc[mts_df.classifier_name == classifier]

        sns.set_theme(style="darkgrid")
        colors = ["#FF0B04", "#4374B3"]
        sns.set_palette(sns.color_palette(colors))

        g = sns.FacetGrid(data, row=param, col="variable",hue="frame",height=6, aspect=1.2)
        g= g.map(sns.lineplot, "groups_lengths", "value", ci=95 )
        # g = g.map(sns.lineplot, "groups_lengths", "value", ci=95)

        for axis in g.axes.flat:
            axis.tick_params(labelleft=True,labelbottom=True)
        plt.show()
        # axes = g.fig.axes
        # for ax in axes:
        #     print(ax.title)
        #     ax.plot("groups_lengths", measures+"_before", data=ucr_classifier, color= "#4374B3")
        #     ax.legend([measures+' After TA', measures+' Before TA'])


        # plt.subplots_adjust(wspace=0.1)
        #
        # plt.savefig(classifier +"" +m+""+ param+ ".png")

        # plt.savefig(classifier + ".png")
        # plt.clf()

#concat_results("C:\\Users\\hadas cohen\\Desktop\\mlp")
#results_table_by_dataset_lengths("C:\\Users\\hadas cohen\\Desktop\\raw_data_results.csv","C:\\Users\\hadas cohen\\Desktop\\mlp\\res_equal-frequency_2_1_-1_1_None.csv", "duration")
#duration

def duration_graph(res_df):

    ucr_df = res_df.loc[res_df.archive_name == "UCRArchive_2018"].iloc[:: -1]
    mts_df = res_df.loc[res_df.archive_name == "mts_archive"].iloc[:: -1]
    for classifier in ["mlp"]:  # todo
        ucr_classifier = ucr_df.loc[ucr_df.classifier_name == classifier]
        mts_classifier = mts_df.loc[mts_df.classifier_name == classifier]
        sns.set_theme(style="darkgrid")

        sns.lineplot(x="groups_lengths", y="duration", data=ucr_classifier, ci=95)
        fig = sns.lineplot(x="groups_lengths", y="duration_before", data=ucr_classifier, ci=95)
        fig.set(xlabel='Length', ylabel='Duration')
        fig.legend(['Duration Before TA', 'Duration After TA'])

        plt.savefig(classifier+" duration" + ".png")
        plt.clf()


def create_all_graphs(param):
    #Duration
    methods = ["gradient", "equal-frequency", "equal-width", "sax","td4c-cosine"]
    # for m in methods:
    res_df, merge = set_df_for_graphs_and_tables("C:\\Users\\Shaha\\Desktop\\raw_data_results.csv",
                                                         "C:\\Users\\Shaha\\Desktop\\results_new.csv", "accuracy","equal-width")
    create_graphs(res_df,merge, param,"accuracy","equal-width")


    #duration_graph(res_df)

    #res_df = set_df_for_graphs_and_tables("C:\\Users\\hadas cohen\\Desktop\\raw_data_results.csv","results_new.csv","accuracy" )
    #create_graphs(res_df,param)

create_all_graphs("paa")

# df1 = pd.DataFrame({'Q': np.arange(10),
#                     'M': np.random.randn(10),
#                     'S': np.random.choice([1, 2], 10)})
#
# df2 = pd.DataFrame({'Q': np.arange(10),
#                     'M': np.random.randn(10),
#                     'S': np.random.choice([1, 2], 10)})
#
# data = pd.concat([df1.assign(frame='df1'),
#                   df2.assign(frame='df2')])
#
# g = sns.FacetGrid(data, col="S", hue='frame')
# g.map(sns.lineplot, "Q", "M")
# plt.legend()
#
# plt.show()