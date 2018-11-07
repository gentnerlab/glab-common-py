#!/usr/local/bin/python
import re
import datetime as dt
import os
from behav.loading import load_data_pandas
from socket import gethostname
import warnings

try:
    import simplejson as json
except ImportError:
    import json

from pyoperant.local import DATA_PATH

process_fname = os.path.join(DATA_PATH, "panel_subject_behavior")

box_nums = []
bird_nums = []
processes = []

# read in boxes, birds, and process information
with open(process_fname, 'rt') as f:
    for line in f.readlines():
        # skip comment lines & blank lines
        if line.strip() and not line.startswith("#"):
            split_line = line.split()
            if split_line[1] == "1":  # box enabled
                box_nums.append(int(split_line[0]))
                bird_nums.append(int(split_line[2]))
                processes.append(split_line[-1])

data_folder = '/home/bird/opdat'

write_lines = []

f = open(os.path.join(DATA_PATH, 'all.compassion'), 'w')

f.write("this all.compassion ran on %s at %s\n" % (gethostname(), dt.datetime.now().strftime('%x %X')))

# Now loop through each bird and grab the error info from each summaryDAT file
for (box, bird, proc) in zip(box_nums, bird_nums, processes):

    if proc in ('Lights',):
        continue

    subj = 'B%d' % (bird)
    box_bird_tag = "box %d\t%s\t" % (box, subj)

    configfname = "/home/bird/opdat/%s/config.json" % (subj)

    if os.path.isfile(configfname):
        with open(configfname, 'rb') as config:
            parameters = json.load(config)
    else:
        parameters = ()

    try:
        sched = parameters['light_schedule']
        bird_owner = parameters['experimenter']['name']
        bird_owner_email = parameters['experimenter']['email']
        shaping = parameters['shape'] if 'shape' in parameters else False
    except Exception as e:
        write_lines.append("%s Error reading config.json: %e" % (box_bird_tag, repr(e)))
        continue

    if shaping:
        write_lines.append("%s Owner %s: Is on shape, hopefully %s is really watching them" % (
            box_bird_tag, bird_owner, bird_owner))
        continue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        behav_data = load_data_pandas([subj], data_folder)

    df = behav_data[subj]
    todays_data = df[(df.index.date - dt.datetime.today().date()) == dt.timedelta(days=0)]
    feeder_ops = sum(todays_data['reward'].values)

    (latitude, longitude) = ('32.82', '-117.14')  # San Diego, CA
    fmt = "%H:%M"

    # initialize the feeding schedule
    if sched == 'sun':
        import ephem

        obs = ephem.Observer()
        obs.lat = latitude  # San Diego, CA
        obs.long = longitude
        sun = ephem.Sun()
        sun.compute()
        start = ephem.localtime(obs.previous_rising(sun))
        end = ephem.localtime(obs.next_setting(sun))
    else:
        if len(sched) != 1:
            write_lines.append("%s Error: non standard schedules, fix this script" % (box_bird_tag))
            continue

        start = dt.datetime.combine(dt.date.today(),
                                    dt.datetime.time(dt.datetime.strptime(sched[0][0], fmt)))
        end = dt.datetime.combine(dt.date.today(),
                                  dt.datetime.time(dt.datetime.strptime(sched[0][1], fmt)))

    duration = end - start
    expected_feeds = 100 * (dt.datetime.now() - start).total_seconds() / duration.total_seconds()
    expected_feeds = min(expected_feeds, 100)

    if feeder_ops < expected_feeds:
        write_lines.append("%s Owner %s: Feeds are insufficient. Expected: %d Actual: %d" % (
            box_bird_tag, bird_owner, expected_feeds, feeder_ops))

write_lines = map(lambda x: x + '\n', write_lines)
f.writelines(write_lines)
f.close()
