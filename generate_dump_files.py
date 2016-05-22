import os
import numpy as np
import sys
import pickle
import pandas as pd
import glob
import pyparsing as pyp
import ntpath
import re


def get_periods(path=None, fu=None):
    c_files = glob.glob("{0}/rtts_u*_n5_*.cpp".format("C:/Users/fep/Documents/src/cayssials2016/n5/u90/new"))

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
    path_5 = "C:/Users/fep/Documents/src/cayssials2016/n5/u90/new/results.txt"
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
    rts_data_dump_file = "n5rtsdata.dump"  # rts info
    dump_file = "n5.dump"  # test results

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

        # save DataFrame to file
        with open(dump_file, "wb") as outfile:
            pickle.dump(data, outfile)
    else:
        print("File {0} already exists.".format(dump_file))


if __name__ == '__main__':
    main()
