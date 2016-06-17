import os
import sys
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from argparse import ArgumentParser
from tabulate import tabulate


def analyze_data(data):
        
    def calculate_task_data(row):
        period = row["t"]
        wcet = row["c"]
        
        task_valid_fu = 0
        if (wcet / period) > (((row['fu'] / 100) * 35.0) / 100.0):
            task_valid_fu = 1
            
        # Finalization and start times of the task releases
        starts = row.values[6::2]
        ends = row.values[7::2]

        # If the test has less than 300 samples, print a warning -- and compute only the first 30 samples
        if ends[-1] == 0:
            ends = ends[:30]
            starts = starts[:30]
            print("warning: task {0} of rts {1} with 30 instances.".format(row["task"], row["rts"]))

        # If the task is not schedulable, mark the task-set as non-valid and skip it
        task_schedulable = 0
        for x in range(len(ends)):
            abs_deadline = (x + 1) * period
            if abs_deadline < ends[x]:
                task_schedulable = 1
                break
    
        # The task is schedulable and valid, so compute the results
        jitter_end = []
        jitter_start = []
        jitter_exec = []
        jitter_wcrt = []
        distance_to_d = []
        distance_to_d2 = []

        for idx in range(1, len(ends)):
            jitter_end.append((np.abs((ends[idx] - ends[idx - 1]) - period)) / period)
            jitter_start.append((np.abs((starts[idx] - starts[idx - 1]) - period)) / period)
            jitter_exec.append((np.abs(ends[idx] - starts[idx]) - wcet) / wcet)
            jitter_wcrt.append((np.abs(ends[idx] - ((idx - 1) * period))))                    
            
        for idx in range(len(ends)):
            abs_deadline = (idx + 1) * period
            distance_to_d.append(np.abs(abs_deadline - ends[idx]) / period)
            distance_to_d2.append(np.abs(ends[idx] - (idx * period)) / period)

        # Add task's results to the task-set result list
        return pd.Series([np.mean(jitter_end), np.mean(jitter_start), np.mean(jitter_exec), np.mean(jitter_wcrt), 
                np.mean(distance_to_d), np.mean(distance_to_d2), wcet / period, task_schedulable, task_valid_fu])

    # Calculate task values and insert as new columns
    data[['jitter_end', 'jitter_start', 'jitter_exec', 'jitter_wcrt', 'distanced', 'distanced2', 'task_fu', 'task_schedulable', 'task_valid_fu']] = data.apply(calculate_task_data, axis=1)
    
    # Filter task-sets that are not schedulable and have valid task uf
    df = data.groupby(['fu', 'rts']).filter(lambda x: (sum(x['task_schedulable']) == 0) and (sum(x['task_valid_fu']) == 0))
    
    return df
    
    
def plot_results(df, prefix):
    # Plots ###
    fus = range(10, 100, 5)
    colors = ["r", "g", "b", "k", "y", "m", "c"]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    #
    # Number of schedulable task-sets
    #
    total_schedulable_rts = df.groupby(['fu']).rts.nunique()
    
    fig, ax = plt.subplots(1, 1)
    total_schedulable_rts.plot(ax=ax, kind="bar")    
    ax.margins(0.5, 0.5)
    ax.set_xlabel('uf')
    ax.set_ylabel('\# of schedulable task-sets')
    #ax.set_xlim([5, 100])    
    #plt.xticks(fus, [str(fu) for fu in fus])
    ax.set_ylim(bottom=0)  # set the bottom value after plotting the values, otherwise it goes from 0 to 1
    plt.savefig("{0}-rts_by_fu.pdf".format(prefix), bbox_inches="tight")
    plt.close(fig)    
    
    #
    # Task mean ending time latency
    #
    fig, ax = plt.subplots(1, 1)
    labels = []
        
    for task, task_group in df.groupby(['task']):
        data = task_group.groupby(['fu'])['jitter_end'].agg({'mean':np.mean})
        labels.append("Task {0}".format(task + 1))
        data.plot(ax=ax)
    
    ax.margins(0.5, 0.5)
    ax.set_xlabel('uf')
    ax.set_ylabel('Mean ending latency')
    ax.set_xlim([5, 100])
    plt.xticks(fus, [str(fu) for fu in fus])    
    ax.set_ylim(bottom=0)
    ax.legend(labels, numpoints=1, loc="best", prop={'size': 9})
    plt.savefig("{0}-end.pdf".format(prefix), bbox_inches="tight")
    plt.close(fig)
    
    #
    # Task mean start time latency
    #
    fig, ax = plt.subplots(1, 1)
    labels = []
        
    for task, task_group in df.groupby(['task']):
        data = task_group.groupby(['fu'])['jitter_start'].agg({'mean':np.mean})
        labels.append("Task {0}".format(task + 1))
        data.plot(ax=ax)
    
    ax.margins(0.5, 0.5)
    ax.set_xlabel('uf')
    ax.set_ylabel('Mean start latency')
    ax.set_xlim([5, 100])
    plt.xticks(fus, [str(fu) for fu in fus])    
    ax.set_ylim(bottom=0)
    ax.legend(labels, numpoints=1, loc="best", prop={'size': 9})
    plt.savefig("{0}-start.pdf".format(prefix), bbox_inches="tight")
    plt.close(fig)
    
    #
    # Task mean execution time
    #
    fig, ax = plt.subplots(1, 1)
    labels = []
        
    for task, task_group in df.groupby(['task']):
        data = task_group.groupby(['fu'])['jitter_exec'].agg({'mean':np.mean})
        labels.append("Task {0}".format(task + 1))
        data.plot(ax=ax)
    
    ax.margins(0.5, 0.5)
    ax.set_xlabel('uf')
    ax.set_ylabel('Mean execution time')
    ax.set_xlim([5, 100])
    plt.xticks(fus, [str(fu) for fu in fus])    
    ax.set_ylim(bottom=0)
    ax.legend(labels, numpoints=1, loc="best", prop={'size': 9})
    plt.savefig("{0}-exec.pdf".format(prefix), bbox_inches="tight")
    plt.close(fig)
    
    #
    # Task mean wcrt time
    #
    fig, ax = plt.subplots(1, 1)
    labels = []
        
    for task, task_group in df.groupby(['task']):
        data = task_group.groupby(['fu'])['jitter_wcrt'].agg({'mean':np.mean})
        labels.append("Task {0}".format(task + 1))
        data.plot(ax=ax)
    
    ax.margins(0.5, 0.5)
    ax.set_xlabel('uf')
    ax.set_ylabel('Mean start latency')
    ax.set_xlim([5, 100])
    plt.xticks(fus, [str(fu) for fu in fus])    
    ax.set_ylim(bottom=0)
    ax.legend(labels, numpoints=1, loc="best", prop={'size': 9})
    plt.savefig("{0}-wcrt.pdf".format(prefix), bbox_inches="tight")
    plt.close(fig)
    
    #
    # Task distance to deadline
    #
    fig, ax = plt.subplots(1, 1)
    labels = []
        
    for task, task_group in df.groupby(['task']):
        data = task_group.groupby(['fu'])['distanced2'].agg({'mean':np.mean})
        labels.append("Task {0}".format(task + 1))
        data.plot(ax=ax)
        
    ax.margins(0.5, 0.5)
    ax.set_xlabel('uf')
    ax.set_ylabel('Mean start latency')
    ax.set_xlim([5, 100])
    ax.set_ylim(bottom=0)
    ax.legend(labels, numpoints=1, loc="best", prop={'size': 9})
    plt.xticks(fus, [str(fu) for fu in fus])
    plt.savefig("{0}-distanced2.pdf".format(prefix), bbox_inches="tight")
    plt.close(fig)
    
    #
    # Task distance to deadline -- one page per task
    #
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages("{0}-distanced2-tasks.pdf".format(prefix)) as pdf:            
        for task, task_group in df.groupby(['task']):
            plt.figure()
            
            data = task_group.groupby(['fu'])['distanced2'].agg({'mean':np.mean, 'max':np.max, 'min':np.min, 'std':np.std})
            data.plot(ax=plt.gca(), y=['mean'], color="b", yerr="std")
            data.plot(ax=plt.gca(), y=['max'], color="r")
            data.plot(ax=plt.gca(), y=['min'], color="g")
            
            plt.margins(0.5, 0.5)
            plt.xlabel('uf')
            plt.ylabel('instance finalization')
            plt.xlim([5, 100])
            plt.ylim([0, 1])
            plt.xticks(fus, [str(fu) for fu in fus])
            plt.title('Task {0}'.format(task + 1))
            plt.legend(['Mean','Max','Min'], numpoints=1, loc="best", prop={'size': 9})            
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close() 
    
    #
    # Utilization factors per task
    #    
    with PdfPages("{0}-fus.pdf".format(prefix)) as pdf:
        for task, task_group in df.groupby(['task']):
            plt.figure()
            
            data = task_group.groupby(['fu'])['task_fu'].agg({'mean':np.mean, 'max':np.max, 'min':np.min, 'std':np.std})
            data.plot(ax=plt.gca(), y=['mean'], color="b", yerr="std")
            data.plot(ax=plt.gca(), y=['max'], color="r")
            data.plot(ax=plt.gca(), y=['min'], color="g")
            
            plt.margins(0.5, 0.5)
            plt.xlabel('rts uf (\%)')
            plt.ylabel('task uf (\%)')
            plt.xlim([5, 100])
            plt.xticks(fus, [str(fu) for fu in fus])
            plt.title('Task {0}'.format(task + 1))
            plt.ylim(bottom=0)
            plt.legend(['Mean','Max','Min'], numpoints=1, loc="best", prop={'size': 9})            
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
                    

def get_args():
    """ Command line arguments """
    parser = ArgumentParser(description="Process data file")
    parser.add_argument("dumpfile", help="Data file", type=str)
    parser.add_argument("prefix", help="prefix for generated files", type=str)
    parser.add_argument("--dumpfiles", nargs="+", help="Dump files", type=str)
    return parser.parse_args()


def main():
    args = get_args()

    if args.dumpfiles:
        if os.path.isfile(args.dumpfile):
            print("{0}: File already exists.".format(args.dumpfile))
            sys.exit(1)
    
        dump_files_dataframes = []
        
        for dump_file in [dump_file for dump_file in args.dumpfiles]:
            if os.path.isfile(dump_file):
                with open(dump_file, "rb") as infile:
                    dump_files_dataframes.append(pickle.load(infile))
            else:
                print("{0}: file not found.".format(dump_file))

        data = pd.concat(dump_files_dataframes)
        df = analyze_data(data)
        
        with open(args.dumpfile, "wb") as outfile:
            pickle.dump(df, outfile)
    else:
        if not os.path.isfile(args.dumpfile):
            print("{0}: File not found.".format(args.dump_file))
            sys.exit(1)
        with open(args.dumpfile, "rb") as infile:
            df = pickle.load(infile)
    
    plot_results(df, args.prefix)


if __name__ == '__main__':
    main()

