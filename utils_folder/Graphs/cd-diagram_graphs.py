import numpy as np
import pandas as pd

from utils_folder.Graphs.ranking_graph import draw_cd_diagram
from utils_folder.Graphs.graphs import get_best_df_after_ta, concat_after_df_with_best_df


def concat_after_df_with_best_df(after_ta_df, best_params, metrics, group_by_lst, number_of_cols):
    dict_after_ta = {k: np.mean for k in metrics}
    best_params = best_params.iloc[:, 0:number_of_cols]
    after_ta_df = after_ta_df.groupby(group_by_lst, as_index=False).agg(dict_after_ta)
    keys = list(best_params.columns.values)
    i1 = after_ta_df.set_index(keys).index
    i2 = best_params.set_index(keys).index
    df = after_ta_df[i1.isin(i2)]
    return df


def create_diagram_classifier(raw_data_df, df_after_ta, type, metric):
    raw_data_df.columns = raw_data_df.columns.str.replace('_raw_data', '')
    raw_data_df = raw_data_df.groupby(["classifier_name", "dataset_name"], as_index=False).agg({metric: np.mean})

    draw_cd_diagram(type + "/CD-Diagram/RawData - Classifier - " + metric, metric, df_perf=raw_data_df,
                    title='Rank Classifiers by ' + metric + ' (Raw Data)', labels=True)

    df_after_ta = df_after_ta.groupby(["classifier_name", "dataset_name"], as_index=False).agg({metric: np.mean})

    draw_cd_diagram(type + "/CD-Diagram/TA - Classifier - " + metric, metric, df_perf=df_after_ta,
                    title='Rank Classifiers by ' + metric + ' (TA)', labels=True)


def create_diagram_combination(raw_data_df, df_after_ta, type, metric):
    metrics = ['MCC', 'Cohen Kappa', 'F1 Score Macro', 'F1 Score Micro', 'F1 Score Weighted', 'Balanced Accuracy',
               'AUC - ROC']
    df_top_from_raw, df_top_ta = get_best_df_after_ta(raw_data_df, df_after_ta, metrics, ["nb bins", "method",
                                                                                          "transformation_type"], 5)

    df_after_ta = concat_after_df_with_best_df(df_after_ta, df_top_ta, metrics,
                                               ["nb bins", "method", "transformation_type", 'dataset_name'],
                                               number_of_cols=3)

    raw_data_df.columns = raw_data_df.columns.str.replace('_raw_data', '')
    df_after_ta['method'] = df_after_ta.apply(
        lambda row: row.method + "_" + str(row["nb bins"]) + "_" + row["transformation_type"], axis=1)

    raw_data_df["method"] = "RawData"
    raw_data_df = raw_data_df.groupby(["method", "dataset_name"], as_index=False).agg({metric: np.mean})
    df_after_ta = df_after_ta.groupby(["method", "dataset_name"], as_index=False).agg({metric: np.mean})

    df = pd.concat([raw_data_df, df_after_ta])

    df = df.rename(columns={"method": "classifier_name"})

    # We deleted this dataset in the middle of the experiments
    if type == "MTS":
        df = df.loc[df['dataset_name'] != "WalkvsRun"]

    draw_cd_diagram(type + "/CD-Diagram/Combination - " + metric, metric, df_perf=df,
                    title='Rank Methods by ' + metric, labels=True)


def create_diagram_combination_without_transformation(raw_data_df, df_after_ta, type, metric):
    metrics = ['MCC', 'Cohen Kappa', 'F1 Score Macro', 'F1 Score Micro', 'F1 Score Weighted', 'Balanced Accuracy',
               'AUC - ROC']
    df_top_from_raw, df_top_ta = get_best_df_after_ta(raw_data_df, df_after_ta, metrics, ["nb bins", "method"], 5)

    df_after_ta = concat_after_df_with_best_df(df_after_ta, df_top_ta, metrics,
                                               ["nb bins", "method", 'dataset_name'],
                                               number_of_cols=2)

    raw_data_df.columns = raw_data_df.columns.str.replace('_raw_data', '')
    df_after_ta['method'] = df_after_ta.apply(lambda row: row.method + "_" + str(row["nb bins"]), axis=1)

    df_after_ta = df_after_ta.groupby(["method", "dataset_name"], as_index=False).agg({metric: np.mean})

    df = df_after_ta.rename(columns={"method": "classifier_name"})

    # We deleted this dataset in the middle of the experiments
    if type == "MTS":
        df = df.loc[df['dataset_name'] != "WalkvsRun"]

    draw_cd_diagram(type + "/CD-Diagram/Combination (Method & Bins) - " + metric, metric, df_perf=df,
                    title='Rank Methods by ' + metric, labels=True)


def create_diagram_method(df_after_ta, type, metric):
    df = df_after_ta.groupby(["method", "dataset_name"], as_index=False).agg({metric: np.mean})

    df = df.rename(columns={"method": "classifier_name"})

    draw_cd_diagram(type + "/CD-Diagram/Method - " + metric, metric, df_perf=df,
                    title='Rank Methods by ' + metric, labels=True)


def create_diagram_transformation(df_after_ta, type, metric):
    df_after_ta = df_after_ta.groupby(["transformation_type", "dataset_name"], as_index=False). \
        agg({metric: np.mean})
    df = df_after_ta.rename(columns={"transformation_type": "classifier_name"})

    draw_cd_diagram(type + "/CD-Diagram/Transformation - " + metric, metric, df_perf=df,
                    title='Rank Transformation Type by ' + metric, labels=True)


def create_diagram_main(type, metric):
    df_after_ta = pd.read_csv("results_of_all_combinations_" + type + "_new.csv", encoding="utf-8")
    raw_data_df = pd.read_csv("Raw_data_results_" + type + "_new.csv", encoding="utf-8")

    # We deleted this dataset in the middle of the experiments
    if type == "MTS":
        df_after_ta = df_after_ta.loc[df_after_ta['dataset_name'] != "WalkvsRun"]
        raw_data_df = raw_data_df.loc[raw_data_df['dataset_name_raw_data'] != "WalkvsRun"]

    # create_diagram_combination(raw_data_df, df_after_ta, type, metric)
    # create_diagram_combination_without_transformation(raw_data_df, df_after_ta, type, metric)
    # create_diagram_method(df_after_ta, type, metric)
    # create_diagram_classifier(raw_data_df, df_after_ta, type, metric)
    create_diagram_transformation(df_after_ta, type, metric)


if __name__ == '__main__':
    create_diagram_main(type="MTS", metric="AUC - ROC")
    create_diagram_main(type="MTS", metric="Balanced Accuracy")

    create_diagram_main(type="UCR", metric="AUC - ROC")
    create_diagram_main(type="UCR", metric="Balanced Accuracy")
