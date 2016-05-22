import os
import sys
import pickle
import pandas as pd


def main():
    dump_file = "n5.dump"

    # verify that the files exists
    if not os.path.isfile(dump_file):
        print("{0}: file not found.".format(dump_file))
        sys.exit(1)

    with open(dump_file, "rb") as infile:
        data = pickle.load(infile)

    results = {}
    results2 = []

    # Group by fu
    for fu, fu_group in data.groupby(["fu"]):

        print("Processing U{0}...".format(fu))

        results[fu] = [0, 0]  # schedulable, not schedulable

        # Group by rts
        for rts, rts_group in fu_group.groupby(["rts"]):

            rts_sched = 0

            # Now group by task
            for task, task_group in rts_group.groupby(["task"]):
                # Finalization time for the instances
                mbed_samples_end = task_group.values[0][6:][1::2]

                for x in range(len(mbed_samples_end)):
                    abs_deadline = (x + 1) * task_group.values[0][4]
                    if abs_deadline < mbed_samples_end[x]:
                        print("Task {0} is not schedulable for RTS {1}, FU {2}: {3} / {4}".format(task, rts, fu, mbed_samples_end[x], abs_deadline))
                        rts_sched = 1
                        break

            results[fu][rts_sched] += 1

    print(results)


if __name__ == '__main__':
    main()
