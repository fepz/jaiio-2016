import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tabulate import tabulate


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


def analyze_data(data):
    results_ends = {}
    results_starts = {}
    results_exec = {}
    results_wcrt = {}
    results_scheds = {}
    results_distanced = {}

    # Group by task
    for task, task_group in data.groupby(["task"]):

        print("Processing task {0}...".format(task))

        results_ends[task] = {}
        results_starts[task] = {}
        results_exec[task] = {}
        results_wcrt[task] = {}
        results_scheds[task] = {}
        results_distanced[task] = {}

        # Group by fu
        for fu, fu_group in task_group.groupby(["fu"]):

            task_end_jitter = []
            task_start_jitter = []
            task_exec_jitter = []
            task_wcrt = []
            task_scheds = [0, 0]
            task_distanced = []

            # Now group by rts
            for rts, rts_group in fu_group.groupby(["rts"]):
                # period
                period = rts_group.iloc[0]["t"]
                wcet = rts_group.iloc[0]["c"]

                # finalization time for the instances
                starts = rts_group.values[0][6:][::2]
                ends = rts_group.values[0][6:][1::2]

                # if the test has less than 300 samples, print a warning -- and compute only the first 30 samples
                if ends[-1] == 0:
                    ends = ends[:30]
                    starts = starts[:30]
                    print("warning: task {0} of rts {1} with 30 instances.".format(task, rts))

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
                distance_to_d = []

                for idx in range(1, len(ends)):
                    jitter_end.append((np.abs((ends[idx] - ends[idx - 1]) - period)) / period)
                    jitter_start.append((np.abs((starts[idx] - starts[idx - 1]) - period)) / period)
                    jitter_exec.append((np.abs(ends[idx] - starts[idx]) - wcet) / wcet)
                    jitter_wcrt.append((np.abs(ends[idx] - ((idx - 1) * period))))
                    
                for idx in range(len(ends)):
                    abs_deadline = (idx + 1) * period
                    distance_to_d.append(np.abs(abs_deadline - ends[idx]))                

                task_end_jitter.append(np.mean(jitter_end))
                task_start_jitter.append(np.mean(jitter_start))
                task_exec_jitter.append(np.mean(jitter_exec))
                task_wcrt.append(np.mean(jitter_wcrt))
                task_distanced.append(np.mean(distance_to_d))

            results_ends[task][fu] = (np.mean(task_end_jitter), np.max(task_end_jitter), np.min(task_end_jitter), np.std(task_end_jitter))
            results_starts[task][fu] = (np.mean(task_start_jitter), np.max(task_start_jitter), np.min(task_start_jitter), np.std(task_start_jitter))
            results_exec[task][fu] = (np.mean(task_exec_jitter), np.max(task_exec_jitter), np.min(task_exec_jitter), np.std(task_exec_jitter))
            results_wcrt[task][fu] = (np.mean(task_wcrt), np.max(task_wcrt), np.min(task_wcrt), np.std(task_wcrt))
            results_scheds[task][fu] = task_scheds
            results_distanced[task][fu] = (np.mean(task_distanced), np.max(task_distanced), np.min(task_distanced), np.std(task_distanced))

            #if task_scheds[0] >= 1000:
            #    break

    for task, fu_results in results_scheds.items():
        for fu, fur in fu_results.items():
            print(task, fu, fur)                    

    # return dict with results
    return {"ends":results_ends, "starts":results_starts, "exec":results_exec, "wcrt":results_wcrt, "scheds":results_scheds, "distanced":results_distanced}


def graph2(results, fus, colors):
    results_ends = results["ends"]
    results_starts = results["starts"]
    results_exec = results["exec"]
    results_wcrt = results["wcrt"]
    results_scheds = results["scheds"]
    results_distanced = results["distanced"]

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
    ax.set_ylabel('j')
    ax.set_xlim([5, 100])
    plt.xticks(fus, [str(fu) for fu in fus])
    for task, taskr in results_distanced.items():
        fu_list = []
        for fu, fur in sorted(taskr.items(), key=lambda f: f[0]):
            fu_list.append(fur[0])
        ax.plot(fus, fu_list, str(colors[task]), label="Task {0}".format(task))
    ax.legend(numpoints=1, loc="best", prop={'size': 9})
    plt.savefig("test5distanced.pdf", bbox_inches="tight")
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

                schedulable = 0
                for x in range(len(ends)):
                    abs_deadline = (x + 1) * period
                    if abs_deadline < ends[x]:
                        schedulable = 1
                        break

                if schedulable > 0:
                    #task_scheds[1] += 1
                    continue                

                if ends[-1] == 0:
                    ends = ends[:30]
                    starts = starts[:30]

                wcrt_l2 = []
                for idx in range(1, len(ends)):
                    wcrt_l2.append((np.abs(ends[idx] - ((idx - 1) * period))))

                wcet_l.append(wcet)
                wcrt_l.append(np.mean(wcrt_l2))
                wcrt_lmax.append(np.max(wcrt_l2))

            print("Task ", task, len(wcet_l), " -- ", np.mean(wcet_l), np.max(wcet_l), np.std(wcet_l), " -- ", np.mean(wcrt_l), np.max(wcrt_lmax))


def get_args():
    """ Command line arguments """
    parser = ArgumentParser(description="Generate dump file")
    parser.add_argument("dumpfiles", nargs="*", help="Dump files", type=str)
    return parser.parse_args()


def main():
    #dump_file = "n5.dump"  # test results
    args = get_args()

    dump_files_dataframes = []
    
    for dump_file in [dump_file for dump_file in args.dumpfiles]:
        if os.path.isfile(dump_file):
            with open(dump_file, "rb") as infile:
                dump_files_dataframes.append(pickle.load(infile))
        else:
            print("{0}: file not found.".format(dump_file))

    fus = range(10, 100, 5)
    colors = ["r", "g", "b", "k", "y", "m", "c"]

    data = pd.concat(dump_files_dataframes)

    graph2(analyze_data(data), fus, colors)
    #analyze_task(data)


if __name__ == '__main__':
    main()

