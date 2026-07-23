"""
This script tells you a bird's accuracy over the past num_days
"""

import sys

sys.path.insert(0, "/home/bird/code/behav-analysis/")

from behav import loading,utils

bird = sys.argv[1]
num_days = int(sys.argv[2])

def check_performance(bird, num_days):
    for subj1, data1 in loading.load_data_pandas([bird],"/home/bird/opdat/").items():
        if subj1 == bird:
            subj = subj1
            data = data1
            continue
    #subj, data = loading.load_data_pandas([bird],"/home/bird/opdat/").items()[0]
    data['date']=data.index.date
    data_filt = utils.filter_normal_trials(utils.filter_recent_days(data, num_days))
    accuracy = data_filt['correct'].mean()
    print("Bird: "+bird+"\nAccuracy: "+str(accuracy)+"\nOver the past "+str(num_days)+" days.")

check_performance(bird,num_days)
