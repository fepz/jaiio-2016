import os
import sys
import pickle
import pandas as pd
import glob
import pyparsing as pyp
import ntpath
import re
from argparse import ArgumentParser


def get_rts_info(path=None, fu=None):
    c_files = glob.glob("{0}/rtts_u*_n*_*.cpp".format(path))

    macrodef1 = "#define TASK_" + pyp.Word(pyp.nums) + "_PERIOD" + pyp.empty + pyp.restOfLine.setResultsName("value")
    macrodef2 = "#define TASK_" + pyp.Word(pyp.nums) + "_WCET" + pyp.empty + pyp.restOfLine.setResultsName("value")

    result = []

    # Find the tasks periods values in the cpp files
    for file in c_files:

        rts_data = [int(s) for s in re.findall('\d+', ntpath.basename(file))]

        with open(file, 'r') as f:
            line = f.read()
            res1 = macrodef1.scanString(line)
            res2 = macrodef2.scanString(line)
            periods = [int(tokens.value) for tokens, startPos, EndPos in res1]
            wcet = [int(tokens.value) for tokens, startPos, EndPos in res2]

        if not wcet:
            print("Can't find tasks wcet values in file {0}.".format(file))
            continue

        if not periods:
            print("Can't find task period values in file {0}.".format(file))
            continue

        for idx, task in enumerate(zip(periods, wcet)):
            t, c = task[0], task[1]
            result.append([rts_data[0], rts_data[1], rts_data[2], idx, t, c])

    return pd.DataFrame(result, columns=["fu", "taskcnt", "rts", "task", "t", "c"], dtype=int)


def get_mbed_data(result_file):
    # verify that the files exists
    if not os.path.isfile(result_file):
        print("Can't find {0} file.".format(result_file))
        sys.exit(1)

    columns_names = ["fu", "taskcnt", "rts", "task"]
    for i in range(300):
        columns_names.append("s")
        columns_names.append("e")

    # Load the data into a DataFrame
    data = pd.read_csv(result_file, sep='\t', dtype=int)
    data.columns = columns_names

    return data


def get_args():
    """ Command line arguments """
    parser = ArgumentParser(description="Generate dump file")
    parser.add_argument("result_file", help="Result file name", type=str)
    parser.add_argument("cpp_path", help="Directory with C files", type=str)
    parser.add_argument("dump_file", help="Dump file name", type=str)
    return parser.parse_args()


def main():
    args = get_args()
    dump_file = args.dump_file

    if not os.path.isfile(dump_file):
        rts_data_dump_file = os.path.splitext(dump_file)[0] + "_rts_data.dump"

        # Load RTS info (periods and wcets)
        if not os.path.isfile(rts_data_dump_file):
            if os.path.isdir(args.cpp_path):
                rts_info = get_rts_info(args.cpp_path)
            else:
                print("Directory {0} not found.".format(args.cpp_path))

            with open(rts_data_dump_file, "wb") as outfile:
                pickle.dump(rts_info, outfile)
        else:
            with open(rts_data_dump_file, "rb") as outfile:
                rts_info = pickle.load(outfile)

        # Load mbed data results and merge with RTS info
        mbed_data = get_mbed_data(args.result_file)
        data = pd.merge(rts_info, mbed_data, on=["fu", "taskcnt", "rts", "task"])

        # save results into file
        with open(dump_file, "wb") as outfile:
            pickle.dump(data, outfile)
    else:
        print("File {0} already exists.".format(dump_file))


if __name__ == '__main__':
    main()
