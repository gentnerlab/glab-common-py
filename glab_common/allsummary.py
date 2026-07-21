from __future__ import print_function
import re
import datetime as dt
from behav.loading import load_data_pandas
import warnings
import subprocess
import os
import sys

process_fname = "/home/bird/opdat/panel_subject_behavior"

box_nums = []
bird_nums = []
processes = []

with open(process_fname, "rt") as psb_file:
    for line in psb_file.readlines():
        if line.startswith("#") or not line.strip():
            pass  # skip comment lines & blank lines
        else:
            spl_line = line.split()
            if spl_line[1] == "1":  # box enabled
                box_nums.append(spl_line[0])
                bird_nums.append(int(spl_line[2]))
                processes.append(spl_line[4])


# rsync magpis
hostname = os.uname()[1]
if "magpi" in hostname:
    for box_num in box_nums:
        box_hostname = box_num
        rsync_src = "bird@{}:/home/bird/opdat/".format(box_hostname)
        rsync_dst = "/home/bird/opdat/"
        print("Rsync src: {}".format(rsync_src), file=sys.stderr)
        print("Rsync dest: {}".format(rsync_dst), file=sys.stderr)
        rsync_output = subprocess.run(["rsync", "-avhW", "--exclude=Generated_Songs", "--exclude=*stim*", rsync_src, rsync_dst])
        print(rsync_output)

subjects = ["B%d" % (bird_num) for bird_num in bird_nums]
data_folder = "/home/bird/opdat"


def format_outline(box, bird, proc, trials=0, feeds=0, TOs=0, noRs=0,
                    feed_errs=("0", "0", "0", "0"), last_feed="N/A",
                    datediff="(no data)"):
    # Always 9 tab-separated fields (box, bird, behav, trials, feeds, TOs,
    # noRs, FeedErrs, LastFeed) -- website_update_utils.py's
    # get_allsummary_on_server() builds a DataFrame with a hardcoded 9-column
    # list from this file's lines, and raises on any row with a different
    # field count.
    return (
        "%s\tB%d\t %s  \ttrls=%s  \tfeeds=%d  \tTOs=%d  \tnoRs=%d  "
        "\tFeedErrs=(%s,%s,%s,%s)  \tlast @ %s %s\n"
        % (
            box, bird, proc, trials, feeds, TOs, noRs,
            feed_errs[0], feed_errs[1], feed_errs[2], feed_errs[3],
            last_feed, datediff,
        )
    )


with open("/home/bird/all.summary", "w") as as_file:
    as_file.write(
        "this all.summary generated at %s\n" % (dt.datetime.now().strftime("%x %X"))
    )
    as_file.write(
        "FeedErr(won't come up, won't go down, already up, resp during feed)\n"
    )

    # Now loop through each bird and grab the error info from each summaryDAT file
    for (box, bird, proc) in zip(box_nums, bird_nums, processes):
        try:
            # make sure box is a string
            box = str(box)
            if proc in ("shape", "lights", "pylights", "lights.py"):
                as_file.write(format_outline(box, bird, proc, datediff="(non-trial box)"))
            else:
                summaryfname = "/home/bird/opdat/B%d/%d.summaryDAT" % (bird, bird)
                with open(summaryfname, "rt") as sdat:
                    sdata = sdat.read()

                m = re.search(r"failures today: (\w+)", sdata)
                hopper_failures = m.group(1)
                m = re.search(r"down failures today: (\w+)", sdata)
                godown_failures = m.group(1)
                m = re.search(r"up failures today: (\w+)", sdata)
                goup_failures = m.group(1)
                m = re.search(r"Responses during feed: (\w+)", sdata)
                resp_feed = m.group(1)

                subj = "B%d" % (bird)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    behav_data = load_data_pandas([subj], data_folder)
                df = behav_data[subj]
                # df = df[~pd.isnull(data.index)]
                todays_data = df[
                    (df.index.date - dt.datetime.today().date()) == dt.timedelta(days=0)
                ]
                feeder_ops = sum(todays_data["reward"].values)
                trials_run = len(todays_data)
                noRs = sum(todays_data["response"].values == "none")
                TOs = trials_run - feeder_ops - noRs
                last_trial_time = todays_data.sort_index().tail().index[-1]
                if last_trial_time.day != dt.datetime.now().day:
                    datediff = "(not today)"
                else:
                    minutes_ago = (dt.datetime.now() - last_trial_time).seconds / 60
                    datediff = "(%d mins ago)" % (minutes_ago)

                outline = format_outline(
                    box, bird, proc,
                    trials=trials_run,
                    feeds=feeder_ops,
                    TOs=TOs,
                    noRs=noRs,
                    feed_errs=(hopper_failures, godown_failures, goup_failures, resp_feed),
                    last_feed=last_trial_time.strftime("%x %X"),
                    datediff=datediff,
                )
                as_file.write(outline)
        except Exception as e:
            as_file.write(format_outline(box, bird, proc, datediff="(error, see allsummary.log)"))
            print(e)
