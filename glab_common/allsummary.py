#!/usr/local/bin/python
import os
import re
import datetime as dt
from behav.loading import load_data_pandas
import warnings
from pyoperant.local import DATA_PATH

process_fname = DATA_PATH + "panel_subject_behavior"

box_nums = []
bird_nums = []
processes = []

# read in boxes, birds, and process information
with open(process_fname, 'rt') as inf:
    for line in inf.readlines():
        # skip comment lines & blank lines
        if line.strip() and not line.startswith("#"):
            spl_line = line.split()
            if spl_line[1] == "1":  # box enabled
                box_nums.append(int(spl_line[0]))
                bird_nums.append(int(spl_line[2]))
                processes.append(spl_line[4])

data_folder = '/home/bird/opdat'

write_lines = []

f = open('/home/bird/all.summary', 'w')

f.write("this all.summary generated at %s\n" % (dt.datetime.now().strftime('%x %X')))
f.write("FeedErr(won't come up, won't go down, already up, resp during feed)\n")

# Now loop through each bird and grab the error info from each summaryDAT file
for (box, bird, proc) in zip(box_nums, bird_nums, processes):

    subj = 'B%d' % (bird)

    box_bird_tag = "box %d\tB%d\t" % (box, bird)

    if proc in ('shape', 'lights', 'pylights', 'lights.py'):
        write_lines.append("%s %s" % (box_bird_tag, proc))
        continue

    summaryfname = "/home/bird/opdat/%s/%d.summaryDAT" % (subj, bird)

    if not os.path.isfile(summaryfname):
        write_lines.append("%s SummaryDat does not exist" % (box_bird_tag))
        continue

    sdata = ""
    with open(summaryfname, 'rt') as f:
        sdata = f.read()

    # m = re.search(r"Trials this session: (\w+)", sdata)
    # trials_run = m.group(1)
    # m = re.search(r"ops today: (\w+)", sdata)
    # feeder_ops = m.group(1)
    # m = re.search(r"d responses: (\w+)", sdata)
    # rf_resp = m.group(1)

    m = re.search(r"failures today: (\w+)", sdata)
    hopper_failures = m.group(1)
    m = re.search(r"down failures today: (\w+)", sdata)
    godown_failures = m.group(1)
    m = re.search(r"up failures today: (\w+)", sdata)
    goup_failures = m.group(1)
    m = re.search(r"Responses during feed: (\w+)", sdata)
    resp_feed = m.group(1)

    # m = re.search(r"Last trial run @: (\w+)\s+(\w+)\s+(\d+)\s+(\d+)\:(\d+)\:(\d+)", sdata)
    # last_trial_day = int(m.group(3))
    # last_trial_hour = int(m.group(4))
    # last_trial_min = int(m.group(5))
    # last_trial_month = m.group(2)

    # # Figure out time since last trial
    # curr_time = datetime.now()
    # if curr_time.day != last_trial_day:
    #     datediff = '(not today)'
    # else:
    #     timediff = (curr_time.hour - last_trial_hour)*60 + (curr_time.minute - last_trial_min)
    #     datediff = '(%d mins ago)' % timediff

    # calculate output fields not written directly
    # TOs = int(rf_resp) - int(feeder_ops)
    # noRs = int(trials_run) - int(rf_resp)

    if hopper_failures is None or godown_failures is None or goup_failures is None or resp_feed is None:
        write_lines.append("%s Error reading SummaryDat, no hopper, go down, go up, or feeding response found" %
                           (box_bird_tag))
        continue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        behav_data = load_data_pandas([subj], data_folder)

    df = behav_data[subj]
    # df = df[~pd.isnull(data.index)]
    todays_data = df[(df.index.date - dt.datetime.today().date()) == dt.timedelta(days=0)]
    feeder_ops = sum(todays_data['reward'].values)
    trials_run = len(todays_data)
    noRs = sum(todays_data['response'].values == 'none')
    TOs = trials_run - feeder_ops - noRs
    last_trial_time = todays_data.sort_index().tail().index[-1]

    if last_trial_time.day != dt.datetime.now().day:
        datediff = '(not today)'
    else:
        minutes_ago = (dt.datetime.now() - last_trial_time).seconds / 60
        datediff = '(%d mins ago)' % (minutes_ago)

    outline = "%s %s  \ttrls=%s  \tfeeds=%d  \tTOs=%d  \tnoRs=%d  \tFeedErrs=(%s,%s,%s,%s)  \tlast @ %s %s" % (
    box_bird_tag, proc, trials_run, feeder_ops, TOs, noRs, hopper_failures, godown_failures, goup_failures, resp_feed,
    last_trial_time.strftime('%x %X'), datediff)

    write_lines.append(outline)

write_lines = map(lambda x: x + '\n', write_lines)
f.writelines(write_lines)
f.close()
