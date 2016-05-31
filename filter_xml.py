import glob
import os
import tempfile
import subprocess
import math
import sys
from argparse import ArgumentParser


def test_rts(max_fu, xmlfile):
    import xml.etree.cElementTree as et

    def get_rts_from_element(elem):
        """ extrae el str de elem """

        def get_int(string):
            """ convierte string a int """
            try:
                return int(string)
            except ValueError:
                return int(float(string))

        rts_id, rts = 0, []
        if elem.tag == 'S':
            rts_id = get_int(elem.get("count"))
            for t in elem.iter("i"):
                task = t.attrib
                for k, v in task.items():
                    task[k] = get_int(v)
                rts.append(task)

        return rts_id, rts

    #context = et.iterparse(xmlfile, events=('start','end',))
    context = et.iterparse(xmlfile, events=('start',))
    context = iter(context)
    event, root = context.__next__()
    #event, root = context.next()

    results = [0, 0, 0, 0]   # sched, non-sched, valid, invalid

    for event, elem in context:
        rts_id, rts = get_rts_from_element(elem)

        if rts:
            result = joseph_wcrt(rts)

            if result[0]:
                # check fu
                results[0] += 1
                fus = [task["C"] / task["T"] for task in rts]
                fu = sum(fus)
                max_percent = (fu * max_fu) / 100
                fu_sel = 2
                for task_fu in fus:
                    if task_fu > max_percent:
                        fu_sel = 3
                        break;
                results[fu_sel] += 1
            else:
                results[1] += 1

        root.clear()
    del context

    return results


def joseph_wcrt(rts):
    """ Verify schedulability """
    wcrt = [0] * len(rts)
    schedulable = True

    wcrt[0] = rts[0]["C"]
    for i, task in enumerate(rts[1:], 1):
        r = 1
        c, t, d = task["C"], task["T"], task["D"]
        while schedulable:
            w = 0
            for taskp in rts[:i]:
                cp, tp = taskp["C"], taskp["T"]
                w += math.ceil(r / tp) * cp
            w = c + w
            if r == w:
                break
            r = w
            if r > d:
                schedulable = False
        wcrt[i] = r
        if not schedulable: break
    return [schedulable, wcrt]


def get_args():
    """ Command line arguments """
    parser = ArgumentParser(description="Filter XML files")
    parser.add_argument("file", help="XML file with RTS", type=str)
    parser.add_argument("fu", help="FU upper bound per task", type=int, default=30)
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.isfile(args.file):
        print("{0}: file not found.".format(args.file))
        sys.exit(1)

    if args.fu <= 0 or args.fu > 100:
        print("Invalid fu: {0}.".format(args.fu))
        sys.exit(1)

    results = test_rts(30, args.file)
    print("{0}: {1}".format(args.file, results))


if __name__ == '__main__':
    main()
