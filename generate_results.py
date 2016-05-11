from __future__ import print_function
import os
import numpy as np
import sys
import pickle
import pandas as pd
import glob
import pyparsing as pyp
import ntpath
import re
import matplotlib.pyplot as plt


def get_periods(path=None, fu=None):
    c_files = glob.glob("{0}/rtts_u*_n5_*.cpp".format("C:/Users/fep/Documents/src/cayssials2016/n5/total/c_files"))

    macrodef1 = "#define TASK_" + pyp.Word(pyp.nums) + "_PERIOD" + pyp.empty + pyp.restOfLine.setResultsName("value")
    macrodef2 = "#define TASK_" + pyp.Word(pyp.nums) + "_WCET" + pyp.empty + pyp.restOfLine.setResultsName("value")

    result = []

    # Find the tasks periods values in the cpp files
    for file in c_files:

        rts_data = [int(s) for s in re.findall('\d+', ntpath.basename(file))]

        with open(file, 'r') as f:
            print("Processing {0}".format(file))
            line = f.read()
            res1 = macrodef1.scanString(line)
            res2 = macrodef2.scanString(line)
            periods = [int(tokens.value) for tokens, startPos, EndPos in res1]
            wcet = [int(tokens.value) for tokens, startPos, EndPos in res2]

        if not wcet:
            print("Can't find tasks wcet values in file {0}.".format(file))
            continue

        for id, task in enumerate(zip(periods, wcet)):
            t, c = task[0], task[1]
            result.append([rts_data[0], rts_data[1], rts_data[2], id, t, c])

    return pd.DataFrame(result, columns=["fu", "taskcnt", "rts", "task", "t", "c"], dtype=int)


def get_mbedata():
    # Files with mbed samples data
    path_5 = "C:\\Users\\fep\\Documents\\src\\cayssials2016\\n5\\total\\results-total.txt"
    path_7 = "C:\\Users\\fep\\Documents\\src\\cayssials2016\\n7\\total\\resultados_total.txt"
    path_10 = "C:\\Users\\fep\\Documents\\src\\cayssials2016\\n10\\total\\resultados_n10_total.txt"
    paths = [("n5", path_5)]  # , ("n7", path_7), ("n10", path_10)]

    #rts_info = get_periods()

    # verify that the files exists
    for _, path in paths:
        if not os.path.isfile(path):
            print("Can't find {0} file.".format(path))
            sys.exit(1)

    columns_names = ["fu", "taskcnt", "rts", "task"]
    for i in range(300):
        columns_names.append("s")
        columns_names.append("e")

    for idx, path in paths:
        # Load the data into a DataFrame
        data = pd.read_csv(path, sep='\t', dtype=int)
        data.columns = columns_names

    return data


def main():
    rts_data_dump_file = "n5rtsdata.dump"
    dump_file = "n5.dump"

    if not os.path.isfile(dump_file):
        if not os.path.isfile(rts_data_dump_file):
            rts_info = get_periods()
            with open(rts_data_dump_file, "wb") as outfile:
                pickle.dump(rts_info, outfile)
        else:
            with open(rts_data_dump_file, "rb") as outfile:
                rts_info = pickle.load(outfile)

        mbed_data = get_mbedata()
        data = pd.merge(rts_info, mbed_data, on=["fu", "taskcnt", "rts", "task"])

        with open(dump_file, "wb") as outfile:
            # save DataFrame to file
            pickle.dump(data, outfile)
    else:
        with open(dump_file, "rb") as infile:
            data = pickle.load(infile)

    data2 = pd.melt(data, id_vars=["fu", "taskcnt", "rts", "task", "t", "c"])

    # por cada str, sacar media, max y min, y dividir por el periodo de la tarea
    # luego, sacar media de los promedios -- y max, min (ya estarian normalizados)

    rts = data2[(data2["rts"] == 3) & (data2["fu"] == 90) & (data2["task"] == 4) & (data2["variable"] == "e")]
    period = rts.iloc[0]["t"]
    ends = rts["value"].values
    print(ends)
    jitter = []
    for idx in range(1, len(ends)):
        jitter.append(ends[idx] - ends[idx-1])
    print(jitter)
    print(period, np.mean(jitter), np.max(jitter), np.min(jitter))

    ########

    # results = []
    # results2 = []
    # for rts, rts_group in data2[(data2["fu"] == 90) & (data2["task"] == 6) & (data2["variable"] == "e")].groupby(["rts"]):
    #     print("-- {0} --".format(rts))
    #     period = rts_group.iloc[0]["t"]
    #     ends = rts_group["value"].values
    #     print(ends)
    #     jitter = []
    #     for idx in range(1, len(ends)):
    #         jitter.append((ends[idx] - ends[idx - 1]))
    #     print(jitter)
    #     print(period, np.mean(jitter), np.max(jitter), np.min(jitter))
    #     jitter /= period
    #     print(jitter)
    #     print(period, np.mean(jitter), np.max(jitter), np.min(jitter))
    #     results.append(np.mean(jitter))
    # print(len(results))
    # print(np.mean(results))

    ########

    fus = range(10, 100, 5)
    colors = ["r", "g", "b", "k", "y", "m", "c"]
    #
    # fig, ax = plt.subplots(1, 1)
    # ax.margins(0.5, 0.5)
    # ax.set_xlabel('fu')
    # ax.set_ylabel('j')
    # ax.set_xlim([0, 100])
    # plt.xticks(fus, [str(fu) for fu in fus])
    #
    # for task, task_group in data2.groupby(["task"]):
    #     means = []
    #
    #     for fu, fu_group in task_group.groupby(["fu"]):
    #         ends = fu_group[(fu_group["variable"] == "e")]["value"].values
    #         jitter = []
    #         for idx in range(1, len(ends)):
    #             jitter.append(ends[idx] - ends[idx - 1])
    #         means.append(np.mean(jitter))
    #
    #     ax.plot(fus, means, str(colors[task]), label=str(task))
    #
    # ax.legend(numpoints=1, loc="best", prop={'size': 9})
    # plt.savefig("test2.png", bbox_inches="tight")
    # plt.close(fig)

    ######

    # fig, ax = plt.subplots(1, 1)
    # ax.margins(0.5, 0.5)
    # ax.set_xlabel('fu')
    # ax.set_ylabel('j')
    # ax.set_xlim([0, 100])
    # plt.xticks(fus, [str(fu) for fu in fus])
    #
    # for task, task_group in data2.groupby(["task"]):
    #     means = []
    #
    #     for fu, fu_group in task_group.groupby(["fu"]):
    #
    #         mean_fu = []
    #
    #         for rts, rts_group in fu_group.groupby(["rts"]):
    #             period = rts_group.iloc[0]["t"]
    #
    #             ends = rts_group[(rts_group["variable"] == "e")]["value"].values
    #
    #             jitter = []
    #             for idx in range(1, len(ends)):
    #                 jitter.append((ends[idx] - ends[idx - 1]) / period)
    #
    #             mean_fu.append(np.mean(jitter))
    #
    #         means.append(np.mean(mean_fu) * 100)
    #
    #     ax.plot(fus, means, str(colors[task]), label=str(task))
    #
    # ax.legend(numpoints=1, loc="best", prop={'size': 9})
    # plt.savefig("test3.png", bbox_inches="tight")
    # plt.close(fig)

    ######

    # fig, ax = plt.subplots(1, 1)
    # ax.margins(0.5, 0.5)
    # ax.set_xlabel('i')
    # ax.set_ylabel('j')
    #
    # for task, task_group in data2[data2["fu"] == 90].groupby(["task"]):
    #
    #     means_i = []
    #
    #     for rts, rts_group in task_group.groupby(["rts"]):
    #         period = rts_group.iloc[0]["t"]
    #
    #         ends = rts_group[(rts_group["variable"] == "e")]["value"].values
    #
    #         jitter = []
    #         for idx in range(1, len(ends)):
    #             jitter.append((ends[idx] - ends[idx - 1]) / period)
    #
    #         means_i.append(np.mean(jitter))
    #
    #     ax.plot(range(100), means_i, str(colors[task]), label=str(task))
    #
    # ax.legend(numpoints=1, loc="best", prop={'size': 9})
    # plt.savefig("test4.png", bbox_inches="tight")
    # plt.close(fig)

    ######

    fig, ax = plt.subplots(1, 1)
    ax.margins(0.5, 0.5)
    ax.set_xlabel('fu')
    ax.set_ylabel('j')
    ax.set_xlim([5, 100])
    ax.set_ylim([0, .25])
    plt.xticks(fus, [str(fu) for fu in fus])

    for task, task_group in data2.groupby(["task"]):
        means = []

        for fu, fu_group in task_group.groupby(["fu"]):

            mean_fu = []

            for rts, rts_group in fu_group.groupby(["rts"]):
                period = rts_group.iloc[0]["t"]

                ends = rts_group[(rts_group["variable"] == "e")]["value"].values

                jitter = []
                for idx in range(1, len(ends)):
                    jitter.append((np.abs((ends[idx] - ends[idx - 1]) - period)) / period)

                mean_fu.append(np.mean(jitter))

            means.append(np.mean(mean_fu))

        ax.plot(fus, means, str(colors[task]), label="Task {0}".format(task))

    ax.legend(numpoints=1, loc="best", prop={'size': 9})
    plt.savefig("test5.pdf", bbox_inches="tight")
    plt.close(fig)

    ######

    ######

    #for task, task_group in data2.groupby(["task"]):
    #    for fu, fu_group in task_group.groupby(["fu"]):
    #        #for var, var_group in fu_group.groupby(["variable"]):
    #        d = fu_group.groupby(["variable"]).agg([np.mean])
    #        print(d)

    fig, ax = plt.subplots(1, 1)
    ax.margins(0.5, 0.5)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Perturbation (\%)')
    ax.set_xlim([0, 100])
    plt.xticks(fus, [str(fu) for fu in fus])

    for fu, fu_group in data2.groupby(["task"]):
        fu_task_group = fu_group.groupby(["fu"])
        d = fu_task_group["t"].agg([np.mean, np.max, np.min])
        ax.plot(fus, d["mean"].values, str(colors[fu]), label=str(fu))

    ax.legend(numpoints=1, loc="best", prop={'size': 9})
    plt.savefig("test.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == '__main__':
    main()

