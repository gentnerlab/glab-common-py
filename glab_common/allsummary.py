from __future__ import print_function
import re
import datetime as dt
from behav.loading import load_data_pandas
import warnings
import subprocess
import os
import sys
import json

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


OPDAT_ROOT = "/home/bird/opdat/"


def get_remote_stim_exclude(box_hostname, bird_num, timeout=8):
    """SSH in and read this subject's config.json to find its real stim_path
    (explicit, or pyoperant's own default of <experiment_path>/stims -- see
    pyoperant.behavior.base.BaseExp.__init__), and turn it into an rsync
    --exclude pattern anchored to opdat/'s root so it only matches this
    subject's own stim dir, not anything else in the tree.

    Returns None if the config can't be read (box unreachable, no config.json
    -- e.g. a lights/shape box) or the subject's stim_path isn't under
    opdat/ at all, so callers can fall back to a generic exclude instead.
    """
    remote_config = "{}B{}/config.json".format(OPDAT_ROOT, bird_num)
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout={}".format(timeout),
             "bird@{}".format(box_hostname), "cat", remote_config],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        config = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None

    experiment_path = config.get("experiment_path") or "{}B{}".format(OPDAT_ROOT, bird_num)
    stim_path = config.get("stim_path") or os.path.join(experiment_path, "stims")

    if stim_path.startswith(OPDAT_ROOT):
        return "/" + stim_path[len(OPDAT_ROOT):]
    return None  # stim_path lives outside opdat/, nothing to exclude here


# rsync magpis
hostname = os.uname()[1]
if "magpi" in hostname:
    for box_hostname, bird_num in zip(box_nums, bird_nums):
        rsync_src = "bird@{}:{}".format(box_hostname, OPDAT_ROOT)
        rsync_dst = OPDAT_ROOT
        print("Rsync src: {}".format(rsync_src), file=sys.stderr)
        print("Rsync dest: {}".format(rsync_dst), file=sys.stderr)

        stim_exclude = get_remote_stim_exclude(box_hostname, bird_num)
        rsync_cmd = ["rsync", "-avhW", "--exclude=Generated_Songs"]
        if stim_exclude:
            rsync_cmd.append("--exclude={}".format(stim_exclude))
        else:
            # Config unreadable (unreachable box, no config.json, etc.) --
            # fall back to the name-based heuristic rather than pulling
            # stimulus libraries unfiltered.
            rsync_cmd.append("--exclude=*stim*")
        rsync_cmd += [rsync_src, rsync_dst]

        rsync_output = subprocess.run(rsync_cmd)
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
