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
from tabulate import tabulate


def get_periods(path=None, fu=None):
    c_files = glob.glob("{0}/rtts_u*_n10_*.cpp".format("C:/Users/fep/Documents/src/cayssials2016/n10/c_files"))

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
    paths = [("n10", path_10)]  # , ("n7", path_7), ("n10", path_10)]

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


def graph1(data, fus):
    data_melted = pd.melt(data, id_vars=["fu", "taskcnt", "rts", "task", "t", "c"])

    fig, ax = plt.subplots(1, 1)
    ax.margins(0.5, 0.5)
    ax.set_xlabel('fu')
    ax.set_ylabel('j')
    ax.set_xlim([5, 100])
    ax.set_ylim([0, .25])
    plt.xticks(fus, [str(fu) for fu in fus])

    result_print = {}

    for task, task_group in data_melted.groupby(["task"]):
        means = []

        result_print[task] = []

        for fu, fu_group in task_group.groupby(["fu"]):

            print("Processing Task {0}, U{1}...".format(task, fu))

            mean_fu = []

            for rts, rts_group in fu_group.groupby(["rts"]):

                if rts == 663:
                    continue  # fault

                period = rts_group.iloc[0]["t"]

                ends = rts_group[(rts_group["variable"] == "e")]["value"].values

                # if there is a missed deadline, use the absolute deadline as end value
                # for x in range(len(ends)):
                #     abs_deadline = (x + 1) * period
                #     if abs_deadline < ends[x]:
                #         ends[x] = abs_deadline

                # if there is a missed deadline, skip this task group
                # for x in range(len(ends)):
                #     abs_deadline = (x + 1) * period
                #     if abs_deadline < ends[x]:
                #         continue

                jitter = []
                for idx in range(1, len(ends)):
                    jitter.append((np.abs((ends[idx] - ends[idx - 1]) - period)) / period)

                mean_fu.append(np.mean(jitter))

            means.append(np.mean(mean_fu))

            # fu, mean, max, min, std
            result_print[task].append((fu, np.mean(mean_fu), np.max(mean_fu), np.min(mean_fu), np.std(mean_fu)))

        # ax.plot(fus, means, str(colors[task]), label="Task {0}".format(task))
        ax.plot(fus, means, "k", label="Task {0}".format(task))

    for k, v in result_print.items():
        print("Task ", k)
        print(tabulate(v, ["fu", "mean", "max", "min", "std"]))

    ax.legend(numpoints=1, loc="best", prop={'size': 9})
    plt.savefig("test5.pdf", bbox_inches="tight")
    plt.close(fig)


def graph2(data, fus, colors):
    results_ends = {}
    results_starts = {}
    results_exec = {}
    results_wcrt = {}
    results_scheds = {}
    #results_faults = {}

    # Group by task
    for task, task_group in data.groupby(["task"]):

        print("Processing task {0}...".format(task))

        results_ends[task] = {}
        results_starts[task] = {}
        results_exec[task] = {}
        results_wcrt[task] = {}
        results_scheds[task] = {}

        # Group by fu
        for fu, fu_group in task_group.groupby(["fu"]):

            task_end_jitter = []
            task_start_jitter = []
            task_exec_jitter = []
            task_wcrt = []
            task_scheds = [0, 0]

            # Now group by rts
            for rts, rts_group in fu_group.groupby(["rts"]):
                # period
                period = rts_group.iloc[0]["t"]
                wcet = rts_group.iloc[0]["c"]

                # finalization time for the instances
                starts = rts_group.values[0][6:][::2]
                ends = rts_group.values[0][6:][1::2]

                if ends[-1] == 0:
                    ends = ends[:30]
                    starts = starts[:30]
                    print("task {0} of rts {1} with 30 instances.".format(task, rts))

                schedulable = 0
                for x in range(len(ends)):
                    abs_deadline = (x + 1) * period
                    if abs_deadline < ends[x]:
                        schedulable = 1
                        break

                if schedulable > 0:
                    task_scheds[1] += 1
                    continue

                task_scheds[0] += 1

                jitter_end = []
                jitter_start = []
                jitter_exec = []
                jitter_wcrt = []

                for idx in range(1, len(ends)):
                    jitter_end.append((np.abs((ends[idx] - ends[idx - 1]) - period)) / period)
                    jitter_start.append((np.abs((starts[idx] - starts[idx - 1]) - period)) / period)
                    jitter_exec.append((np.abs(ends[idx] - starts[idx]) - wcet) / wcet)
                    jitter_wcrt.append((np.abs(ends[idx] - ((idx - 1) * period))))

                task_end_jitter.append(np.mean(jitter_end))
                task_start_jitter.append(np.mean(jitter_start))
                task_exec_jitter.append(np.mean(jitter_exec))
                task_wcrt.append(np.mean(jitter_wcrt))

            results_ends[task][fu] = (np.mean(task_end_jitter), np.max(task_end_jitter), np.min(task_end_jitter),
                                      np.std(task_end_jitter))
            results_starts[task][fu] = (np.mean(task_start_jitter), np.max(task_start_jitter), np.min(task_start_jitter),
                                        np.std(task_start_jitter))
            results_exec[task][fu] = (np.mean(task_exec_jitter), np.max(task_exec_jitter), np.min(task_exec_jitter),
                                        np.std(task_exec_jitter))
            results_wcrt[task][fu] = (np.mean(task_wcrt), np.max(task_wcrt), np.min(task_wcrt), np.std(task_wcrt))
            results_scheds[task][fu] = task_scheds

    fig, ax = plt.subplots(1, 1)
    ax.margins(0.5, 0.5)
    ax.set_xlabel('fu')
    ax.set_ylabel('j')
    ax.set_xlim([5, 100])
    plt.xticks(fus, [str(fu) for fu in fus])
    for task, taskr in results_ends.items():
        fu_list = []
        for fu, fur in sorted(taskr.items(), key=lambda f: f[0]):
            fu_list.append(fur[0])
        ax.plot(fus, fu_list, str(colors[task]), label="Task {0}".format(task))
    ax.legend(numpoints=1, loc="best", prop={'size': 9})
    plt.savefig("test5ends.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.margins(0.5, 0.5)
    ax.set_xlabel('fu')
    ax.set_ylabel('j')
    ax.set_xlim([5, 100])
    plt.xticks(fus, [str(fu) for fu in fus])
    for task, taskr in results_starts.items():
        fu_list = []
        for fu, fur in sorted(taskr.items(), key=lambda f: f[0]):
            fu_list.append(fur[0])
        ax.plot(fus, fu_list, str(colors[task]), label="Task {0}".format(task))
    ax.legend(numpoints=1, loc="best", prop={'size': 9})
    plt.savefig("test5starts.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.margins(0.5, 0.5)
    ax.set_xlabel('fu')
    ax.set_ylabel('j')
    ax.set_xlim([5, 100])
    plt.xticks(fus, [str(fu) for fu in fus])
    for task, taskr in results_exec.items():
        fu_list = []
        for fu, fur in sorted(taskr.items(), key=lambda f: f[0]):
            fu_list.append(fur[0])
        ax.plot(fus, fu_list, str(colors[task]), label="Task {0}".format(task))
    ax.legend(numpoints=1, loc="best", prop={'size': 9})
    plt.savefig("test5execs.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.margins(0.5, 0.5)
    ax.set_xlabel('fu')
    ax.set_ylabel('j')
    ax.set_xlim([5, 100])
    plt.xticks(fus, [str(fu) for fu in fus])
    for task, taskr in results_wcrt.items():
        fu_list = []
        for fu, fur in sorted(taskr.items(), key=lambda f: f[0]):
            fu_list.append(fur[0])
        ax.plot(fus, fu_list, str(colors[task]), label="Task {0}".format(task))
    ax.legend(numpoints=1, loc="best", prop={'size': 9})
    plt.savefig("test5wcrt.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.margins(0.5, 0.5)
    ax.set_xlabel('fu')
    ax.set_ylabel('# of schedulable systems')
    ax.set_xlim([0, 100])
    plt.xticks(fus, [str(fu) for fu in fus])
    for task, taskr in results_scheds.items():
        fu_list = []
        for fu, fu_scheds in sorted(taskr.items(), key=lambda f: f[0]):
            fu_list.append(fu_scheds[0])
        ax.plot(fus, fu_list, str(colors[task]), label="Task {0}".format(task))
    ax.legend(numpoints=1, loc="best", prop={'size': 9})
    plt.savefig("test5scheds.pdf", bbox_inches="tight")
    plt.close(fig)


def analyze_task(data):
    for fu, fu_group in data.groupby(["fu"]):
        print("fu ", fu)
        for task, task_group in fu_group.groupby(["task"]):
            wcet_l = []
            wcrt_l = []
            wcrt_lmax = []
            for rts, rts_group in task_group.groupby(["rts"]):
                # period
                period = rts_group.iloc[0]["t"]
                wcet = rts_group.iloc[0]["c"]

                # finalization time for the instances
                starts = rts_group.values[0][6:][::2]
                ends = rts_group.values[0][6:][1::2]

                if ends[-1] == 0:
                    ends = ends[:30]
                    starts = starts[:30]

                wcrt_l2 = []
                for idx in range(1, len(ends)):
                    wcrt_l2.append((np.abs(ends[idx] - ((idx - 1) * period))))

                wcet_l.append(wcet)
                wcrt_l.append(np.mean(wcrt_l2))
                wcrt_lmax.append(np.max(wcrt_l2))

            print("Task ", task, np.mean(wcet_l), np.max(wcet_l), np.std(wcet_l), " -- ", np.mean(wcrt_l), np.max(wcrt_lmax))


def main():
    rts_data_dump_file = "n5rtsdata.dump"
    dump_file = "n5.dump"
    dump_sched_file = "n5sched.dump"

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

    with open("n5sched.dump", "rb") as infile:
        data_sched = pickle.load(infile)



    #with open("n5melted.dump", "wb") as outfile:
    #    pickle.dump(data2, outfile)

    #with open("n5sched_melt.dump", "wb") as outfile:
    #    pickle.dump(data2, outfile)

    # por cada str, sacar media, max y min, y dividir por el periodo de la tarea
    # luego, sacar media de los promedios -- y max, min (ya estarian normalizados)

    # rts = data2[(data2["rts"] == 663) & (data2["fu"] == 30) & (data2["task"] == 2) & (data2["variable"] == "e")]
    # period = rts.iloc[0]["t"]
    # ends = rts["value"].values
    # print(ends)
    # jitter = []
    # for idx in range(1, len(ends)):
    #     jitter.append((np.abs((ends[idx] - ends[idx - 1]) - period)) / period)
    # print(jitter)
    # print(period, np.mean(jitter), np.max(jitter), np.min(jitter))

    ########

    # results = []
    # max = []
    # for rts, rts_group in data2[(data2["fu"] == 30) & (data2["task"] == 2) & (data2["variable"] == "e")].groupby(["rts"]):
    #     print("-- {0} --".format(rts))
    #     period = rts_group.iloc[0]["t"]
    #     ends = rts_group["value"].values
    #     print(ends)
    #     jitter = []
    #     for idx in range(1, len(ends)):
    #         #jitter.append((ends[idx] - ends[idx - 1]))
    #         #jitter.append(np.abs(ends[idx] - ends[idx - 1]) - period)
    #         jitter.append((np.abs((ends[idx] - ends[idx - 1]) - period)) / period)
    #     #print(jitter)
    #     #print(period, np.mean(jitter), np.max(jitter), np.min(jitter))
    #     #jitter /= period
    #     print(jitter)
    #     print(period, np.mean(jitter), np.max(jitter), np.min(jitter))
    #     results.append(np.mean(jitter))
    #     if np.mean(jitter) > 1:
    #         max.append(rts)
    # print(len(results))
    # print(np.mean(results), np.max(results), np.min(results), np.std(results))
    # print(max)

    ########

    fus = range(10, 100, 5)
    colors = ["r", "g", "b", "k", "y", "m", "c"]

    graph2(data, fus, colors)
    #analyze_task(data)

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



    ######

    ######

    #for task, task_group in data2.groupby(["task"]):
    #    for fu, fu_group in task_group.groupby(["fu"]):
    #        #for var, var_group in fu_group.groupby(["variable"]):
    #        d = fu_group.groupby(["variable"]).agg([np.mean])
    #        print(d)

    # fig, ax = plt.subplots(1, 1)
    # ax.margins(0.5, 0.5)
    # ax.set_xlabel('Frequency')
    # ax.set_ylabel('Perturbation (\%)')
    # ax.set_xlim([0, 100])
    # plt.xticks(fus, [str(fu) for fu in fus])
    #
    # for fu, fu_group in data2.groupby(["task"]):
    #     fu_task_group = fu_group.groupby(["fu"])
    #     d = fu_task_group["t"].agg([np.mean, np.max, np.min])
    #     ax.plot(fus, d["mean"].values, str(colors[fu]), label=str(fu))
    #
    # ax.legend(numpoints=1, loc="best", prop={'size': 9})
    # plt.savefig("test.png", bbox_inches="tight")
    # plt.close(fig)


if __name__ == '__main__':
    main()

