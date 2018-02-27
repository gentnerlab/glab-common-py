#!/usr/local/bin/python
import re
import datetime as dt
from behav.loading import load_data_pandas
from socket import gethostname
import warnings
try:
    import simplejson as json
except ImportError:
    import json

from pyoperant.local import DATA_PATH

process_fname = DATA_PATH + "panel_subject_behavior"

box_nums = []
bird_nums = []
processes = []

with open(process_fname, 'rt') as in_f:
    for line in in_f.readlines():
        if line.startswith('#') or not line.strip():
            pass # skip comment lines & blank lines
        else:
            spl_line = line.split()
            if spl_line[1] == "1": #box enabled
                box_nums.append(int(spl_line[0]))
                bird_nums.append(int(spl_line[2]))
                processes.append(spl_line[-1])

subjects = ['B%d' % (bird_num) for bird_num in bird_nums]
data_folder = '/home/bird/opdat'

with open(DATA_PATH+'all.compassion', 'w') as f:

    f.write("this all.compassion ran on %s at %s\n" % (gethostname(), dt.datetime.now().strftime('%x %X')))

    # Now loop through each bird and grab the error info from each summaryDAT file
    for (box, bird, proc) in zip(box_nums, bird_nums, processes):
        try:
            if proc in ('Lights',):
                pass
            else:
                configfname = "/home/bird/opdat/B%d/config.json" % (bird)
                try:
                    with open(configfname, 'rb') as config:
                        parameters = json.load(config)
                except IOError:
                    parameters = ()

                sched = parameters['light_schedule']
                bird_owner = parameters['experimenter']['name']
                bird_owner_email = parameters['experimenter']['email']
                shaping = parameters['shape'] if 'shape' in parameters else False

                if shaping:
                    f.write("Bird B%d\tBox %d\tOwner %s: Is on shape, hopefully %s is really watching them\n" % (bird, box, bird_owner, bird_owner))
                else:
                    subj = 'B%d' % (bird)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        behav_data = load_data_pandas([subj], data_folder);
                    df = behav_data[subj]
                    todays_data = df[(df.index.date-dt.datetime.today().date()) == dt.timedelta(days=0)]
                    feeder_ops = sum(todays_data['reward'].values)

                    (latitude, longitude) = ('32.82', '-117.14') # San Diego, CA
                    fmt="%H:%M"
                    if sched == 'sun':
                        import ephem
                        obs = ephem.Observer()
                        obs.lat = latitude # San Diego, CA
                        obs.long = longitude
                        sun = ephem.Sun()
                        sun.compute()
                        start = ephem.localtime(obs.previous_rising(sun))
                        end = ephem.localtime(obs.next_setting(sun))
                    else:
                        if len(sched) != 1:
                            raise Exception("someone is using non standard schedules and has to fix this script themselves")
                        start = dt.datetime.combine(dt.date.today(), dt.datetime.time(dt.datetime.strptime(sched[0][0], fmt)))
                        end = dt.datetime.combine(dt.date.today(), dt.datetime.time(dt.datetime.strptime(sched[0][1], fmt)))

                    duration = end - start
                    expected_feeds = 100 * (dt.datetime.now() - start).total_seconds() / duration.total_seconds()
                    expected_feeds = min(expected_feeds, 100)
                    
                    if feeder_ops < expected_feeds:
                        f.write("Bird B%d\tBox %d\tOwner %s: Feeds are insufficient. Expected: %d Actual: %d\n" % (bird, box, bird_owner, expected_feeds, feeder_ops))
               
        except Exception as e:
            f.write("box %d\tB%d\t Error opening SummaryDat or incorrect format\n" % (box, bird))
            print e, box, bird, proc
