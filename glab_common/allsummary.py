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
                # The command field is multi-word (e.g. "behave -P 1 -S <3>
                # Lights"), with the actual behavior/protocol name as its
                # last token -- spl_line[4] would just be the literal
                # string "behave" every time.
                processes.append(spl_line[-1])


OPDAT_ROOT = "/home/bird/opdat/"
STIM_EXCLUDES_FNAME = "/home/bird/opdat/panel_stim_excludes"
SYNC_STATE_FNAME = "/home/bird/opdat/.allsummary_sync_state.json"


def load_stim_excludes(loc=STIM_EXCLUDES_FNAME):
    """Reads rpioperantctl's panel_stim_excludes file (panel, subj, exclude
    -- tab-separated, one row per panel) and returns {panel: exclude}.
    rpioperantctl already SSHes into every panel every 5 min to check
    running processes, and resolves each subject's real stim_path from its
    config.json (explicit, or pyoperant's own <experiment_path>/stims
    default) as part of that same connection -- so allsummary.py just reads
    the result here instead of opening its own SSH connections to redo that
    lookup on every 15-min run.

    Returns {} if the file doesn't exist yet (e.g. rpioperantctl hasn't run
    since this was added) so callers can fall back to a generic exclude.
    """
    excludes = {}
    try:
        with open(loc, "rt") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    panel, subj, stim_exclude = parts
                    excludes[panel] = stim_exclude
    except OSError:
        pass
    return excludes


def load_sync_state(loc=SYNC_STATE_FNAME):
    """Returns {"last_mapping": {box: bird}, "pending_stragglers": [[box, bird], ...]}
    from the previous run, so a bird that gets swapped out of
    panel_subject_behavior between rsync cycles (its box now points at a
    new/no bird) can still get one final catch-up pull instead of being
    silently dropped -- see sync_subject()/the main sync loop below.
    Returns empty defaults if the file doesn't exist yet (first run).
    """
    try:
        with open(loc, "rt") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        state = {}
    state.setdefault("last_mapping", {})
    state.setdefault("pending_stragglers", [])
    return state


def save_sync_state(last_mapping, pending_stragglers, loc=SYNC_STATE_FNAME):
    with open(loc, "wt") as f:
        json.dump(
            {"last_mapping": last_mapping, "pending_stragglers": pending_stragglers},
            f,
        )


def sync_subject(box_hostname, bird_num, stim_exclude=None):
    """rsyncs one subject's own opdat/<subj>/ folder from its box. Returns
    True on success (rsync returncode 0), so callers can tell whether a
    straggler pull actually caught up or still needs retrying next cycle.
    """
    subj = "B%d" % bird_num
    rsync_src = "bird@{}:{}{}/".format(box_hostname, OPDAT_ROOT, subj)
    rsync_dst = "{}{}/".format(OPDAT_ROOT, subj)
    print("Rsync src: {}".format(rsync_src), file=sys.stderr)
    print("Rsync dest: {}".format(rsync_dst), file=sys.stderr)

    # Always exclude by name (*stim*) -- every real stim dir name seen
    # so far (stims, stimuli, cdp_stimuli, stimulus_set, ...) contains
    # "stim", so this is the safety net for the common case; the
    # config-resolved path (when given) is exact-match precision on top
    # of it, for a subject whose stim_path doesn't follow that pattern.
    rsync_cmd = ["rsync", "-avhW", "--exclude=Generated_Songs", "--exclude=*stim*"]
    if stim_exclude:
        rsync_cmd.append("--exclude={}".format(stim_exclude))
    rsync_cmd += [rsync_src, rsync_dst]

    rsync_output = subprocess.run(rsync_cmd)
    print(rsync_output)
    return rsync_output.returncode == 0


# rsync magpis
hostname = os.uname()[1]
if "magpi" in hostname:
    stim_excludes = load_stim_excludes()
    current_mapping = dict(zip(box_nums, bird_nums))

    # Only pull each box's active subject's own folder, not its whole
    # opdat/ tree -- a box accumulates a folder for every subject that's
    # ever run there (confirmed live: magpi20's opdat/ still had 8 other
    # birds' folders alongside its current one), and no client ever runs
    # more than one bird at a time, so those other folders are orphaned
    # and generate no new data. Pulling the whole tree meant re-pulling
    # all of that (plus any box-level shared stim dirs) on every run.
    #
    # But narrowing to just the *current* mapping means a bird that gets
    # swapped out of panel_subject_behavior between rsync cycles (its box
    # now points at a new/no bird) would never get synced again, silently
    # dropping however much of its final session hadn't been pulled yet.
    # So: compare this run's mapping to the previous one, and give any
    # box whose bird changed one more pull of the *old* bird's folder,
    # keeping it queued (retried every cycle) until that pull actually
    # succeeds -- self-healing if the box happened to be unreachable
    # exactly when the swap happened.
    sync_state = load_sync_state()
    pending_stragglers = [tuple(p) for p in sync_state["pending_stragglers"]]
    for box_hostname, old_bird in sync_state["last_mapping"].items():
        if current_mapping.get(box_hostname) != old_bird:
            straggler = (box_hostname, old_bird)
            if straggler not in pending_stragglers:
                pending_stragglers.append(straggler)

    for box_hostname, bird_num in current_mapping.items():
        sync_subject(box_hostname, bird_num, stim_exclude=stim_excludes.get(box_hostname))

    still_pending = []
    for box_hostname, bird_num in pending_stragglers:
        print("Straggler catch-up: {} / B{}".format(box_hostname, bird_num), file=sys.stderr)
        # Not using stim_excludes here -- that's resolved from the box's
        # *current* subject's config.json, which may not match the
        # departed bird's own stim_path. The *stim* glob alone still
        # applies inside sync_subject().
        if not sync_subject(box_hostname, bird_num):
            still_pending.append([box_hostname, bird_num])

    save_sync_state(current_mapping, still_pending)

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
            if proc.lower() in ("shape", "lights", "pylights", "lights.py"):
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
