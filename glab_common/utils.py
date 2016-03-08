import scipy.io as sio
import numpy as np
import glob
import pandas as pd
import datetime as dt
import os
import matplotlib.pyplot as plt

def load_mat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    Parameters:
    -----------
    filename : str
        matlab data file ('.mat')

    Returns:
    --------
    dict
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def load_rDAT(fin,nheaderrows = 0,fmt=None):
    if fmt == None: #replace with your own rdat format
        fmt = [('session','i4'),
               ('trial','i4'),
               ('normal','b'),
               ('stimulus','a64'),
               ('class','i4'),
               ('R_sel','i4'),
               ('R_acc','i4'),
               ('ReactionTime','f4'),
               ('Reinforced','b'),
               ('TimeOfDay','a8'),
               ('Date','a8'),
               ];

    while True:
        if nheaderrows > 100:
            raise ValueError('Recursively found more than 100 header rows.')
        try:
            data = np.genfromtxt(fin,dtype=fmt,invalid_raise=False,skip_header=nheaderrows)
            return data
        except ValueError:
            nheaderrows += 1

def _make_dt_maker(year):
    def dt_maker(x):
        return dt.datetime(year, int(x['old_date'][0:2]), int(x['old_date'][2:]), int(x['TimeOfDay'][0:2]), int(x['TimeOfDay'][2:]))
    return dt_maker

def _read_year_rDAT(rDat_f, nheaderrows):
    with open(rDat_f) as f:
        head = [f.next() for x in xrange(nheaderrows)]
    date_line = filter(lambda x:'Start time' in x, head)
    return int(date_line[0][-5:-1])

def load_data_pandas(subjects, data_folder, force_boolean=['reward']):
    '''
    a function that loads data files for a number of subjects into panda DataFrames.
    supports pyoperant files, rDAT files (from c operant behavior scripts), and AllTrials files

    Parameters:
    -----------
    subjects :  list or tuple of str
        bird ids of any length i.e. ('B999', 'B9999')
    data_folder : str
        top level folder for the data containing folders matching the elements of subjects
    force_boolean : list of str, optional
        data columns which will be cast as bool

    Returns:
    --------
    dict
        each key in the dict is a subject string. each value is a pandas dataframe
    '''
    behav_data = {}
    for subj in subjects:
        df_set = []

        # if vogel/pyoperant
        data_files = glob.glob(os.path.join(data_folder,subj,subj+'_trialdata_*.csv'))
        if data_files:
            for data_f in data_files:
                with open(data_f,'rb') as f:
                    try:
                        df = pd.read_csv(f,index_col=['time'],parse_dates=['time'])
                        df['data_file'] = data_f
                        df_set.append(df)
                    except ValueError:
                        df = None

        # if ndege/c operant
        data_files = glob.glob(os.path.join(data_folder,subj,subj[1:]+'_match2sample*.2ac_rDAT'))
        if data_files:  
            fmt = [('session','i4'),
                   ('trial_number','i4'),
                   ('old_type','b'),
                   ('stimulus','a64'),
                   ('old_class','i4'),
                   ('old_response','i4'),
                   ('old_correct','i4'),
                   ('rt','f4'),
                   ('reinforcement','b'),
                   ('TimeOfDay','a8'),
                   ('old_date','a8'),
                   ];
            for data_f in data_files:
                nheaderrows = 5
                dat = load_rDAT(data_f, nheaderrows=nheaderrows, fmt=fmt)

                year = _read_year_rDAT(data_f, nheaderrows)
                df = pd.DataFrame(dat, columns=zip(*fmt)[0])
                dt_maker = _make_dt_maker(year)
                df['date'] = df.apply(dt_maker, axis=1)
                df.set_index('date', inplace=True)
                df['type_'] = df['old_type'].map(lambda(x): ['correction','normal'][x])
                df['response'] = df['old_response'].map(lambda(x): ['none', 'L', 'R'][x])
                df['correct'] = df['old_correct'].map(lambda(x): [False, True, float('nan')][x])
                df['reward'] = df.apply(lambda(x): x['reinforcement'] == 1 and x['correct'] == True, axis=1)
                df['class_'] = df['old_class'].map(lambda(x): ['none', 'L', 'R'][x])
                df['data_file'] = data_f
                df_set.append(df)

        # if ndege/c GONOGO operant
        data_files = glob.glob(os.path.join(data_folder,subj,subj[1:]+'*.gonogo_rDAT'))
        if data_files:  
            fmt = [('session','i4'),
                   ('trial_number','i4'),
                   ('old_type','b'),
                   ('stimulus','a64'),
                   ('old_class','i4'),
                   ('old_response','i4'),
                   ('old_correct','i4'),
                   ('rt','f4'),
                   ('reinforcement','b'),
                   ('TimeOfDay','a8'),
                   ('old_date','a8'),
                   ];
            for data_f in data_files:
                nheaderrows = 5
                dat = load_rDAT(data_f, nheaderrows=nheaderrows, fmt=fmt)

                year = _read_year_rDAT(data_f, nheaderrows)
                df = pd.DataFrame(dat, columns=zip(*fmt)[0])
                dt_maker = _make_dt_maker(year)
                df['date'] = df.apply(dt_maker, axis=1)
                df.set_index('date', inplace=True)
                df['type_'] = df['old_type'].map(lambda(x): ['correction','normal'][x])
                df['response'] = df['old_response'].map(lambda(x): ['none', 'C'][x])
                df['correct'] = df['old_correct'].map(lambda(x): [False, True, float('nan')][x])
                df['reward'] = df.apply(lambda(x): x['reinforcement'] == 1 and x['correct'] == True, axis=1)
                df['class_'] = df['old_class'].map(lambda(x): ['none', 'GO', 'NOGO'][x])
                df['data_file'] = data_f
                df_set.append(df)
                        
        # if AllTrials file from probe-the-broab
        data_files = glob.glob(os.path.join(data_folder,subj,subj+'.AllTrials'))
        if data_files: 
            col_map = {'StimName': 'stimulus',
                       'Epoch': 'session',
                       'StimulusFile': 'block_name',
                       }
            def _parse(datestr, timestr):
                return dt.datetime.strptime(datestr+timestr,"%Y:%m:%d%H:%M:%S")

            for data_f in data_files:
                nheaderrows = 1
                # try:
                df = pd.read_csv(data_f,
                                 parse_dates={'date':['Date','Time']},
                                 date_parser=_parse,
                                 )
                df.rename(columns=col_map, inplace=True)
                df.set_index('date',inplace=True)
                df['type_'] = df['Correction'].map(lambda(x): {0:'normal',1:'correction',243:'error',-1:None}[x])
                df['correct'] = df['ResponseAccuracy'].map(lambda(x): [False, True, float('nan')][x])
                df['reward'] = df.apply(lambda(x): x['Reinforced'] == 1 and x['correct'] == True, axis=1)
                df['punish'] = df.apply(lambda(x): x['Reinforced'] == 1 and x['correct'] == False, axis=1)
                df['class_'] = df['StimClass'].map(lambda(x): {0:'none',1:'L',2:'R',243:'error',-1:None}[x])
                df['response'] = df['ResponseSelection'].map(lambda(x): ['none', 'L', 'R'][x])
                df['data_file'] = data_f

                is_behave = df['BehavioralRecording'] > 0
                df = df[is_behave]

                df_set.append(df)
                (force_boolean.append(x) for x in ['NeuralRecording','BehavioralRecording'])

                # except ValueError:
                #     df = None
        if df_set:
            behav_data[subj] = pd.concat(df_set).sort()
        else:
            print 'data not found for %s' % (subj)
    if force_boolean:
        for subj in subjects:
            if subj in behav_data:
                for forced in force_boolean:
                    behav_data[subj][forced] = behav_data[subj][forced].map(lambda(x): x in [True, 'True', 'true', 1, '1'])
    return behav_data

def binP(N, p, x1, x2):
    p = float(p)
    q = p/(1-p)
    k = 0.0
    v = 1.0
    s = 0.0
    tot = 0.0

    while(k<=N):
            tot += v
            if(k >= x1 and k <= x2):
                    s += v
            if(tot > 10**30):
                    s = s/10**30
                    tot = tot/10**30
                    v = v/10**30
            k += 1
            v = v*q*(N+1-k)/k
    return s/tot

def binomial_ci(x,N,CL=95.0):
    '''
    Calculate the exact confidence interval for a binomial proportion

    from http://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals

    Parameters:
    -----------
    x : int
        count of items
    N : int
        total number of items
    CL : float
        confidence limit

    Returns:
    --------
    tuple of floats
        the lower and upper bounds on the confidence interval

    Usage:
    >>> calcBin(13,100)
    (0.07107391357421874, 0.21204372406005856)
    >>> calcBin(4,7)
    (0.18405151367187494, 0.9010086059570312)
    '''
    x = float(x)
    N = float(N)
    #Set the confidence bounds
    TU = (100 - float(CL))/2
    TL = TU

    P = x/N
    if (x==0):
        dl = 0.0
    else:
        v = P/2
        sL = 0
        sH = P
        p = TL/100

        while((sH-sL) > 10**-5):
            if(binP(N, v, x, N) > p):
                sH = v
                v = (sL+v)/2
            else:
                sL = v
                v = (v+sH)/2
        dl = v

    if (x==N):
        ul = 1.0
    else:
        v = (1+P)/2
        sL = P
        sH = 1
        p = TU/100
        while((sH-sL) > 10**-5):
            if(binP(N, v, 0, x) < p):
                sH = v
                v = (sL+v)/2
            else:
                sL = v
                v = (v+sH)/2
        ul = v
    return (dl, ul)

def vinjegallant(response):
    '''
    calculates the activity fraction of a set of responses

    Parameters:
    -----------
    response : list or tuple or NumPy array
        the set of responses to calculate the activity fraction over

    Returns:
    --------
    float
    '''
    R = np.asarray(response[:])
    n = np.float_(len(R))
    eps = np.spacing(np.float64(1))

    A = ((R.sum()/n)**2) / (((R**2).sum()/n) + eps)
    S = (1 - A) / (1 - 1/n)

    return S



def accperstimplot(subj,df,days=7,stims_all=None):
    '''
    perc corr broken out by stimulus and day. input how many days you want to look at into "days" argument
    stims_all is a list of all stimuli in the order you want them displayed

    Parameters:
    -----------
    subj : str
        the subject
    df : pandas DataFrame
        the data to analyze
    days : int, optional
        how many of the previous days to observe. (default=7)
    stims_all : list of str
        list of all stimuli in the order you want them displayed
    '''
    data_to_analyze = df[(df.response!='none')&(df.type_=='normal')&(df.index>(dt.datetime.today()-dt.timedelta(days=days)))]
    #get any stims that have been shown to bird
    if not stims_all:
        stims_all = sorted(list(set(data_to_analyze.stimulus)))
    #stims = list(set(data_day.stimulus))
    blocked = data_to_analyze.groupby([lambda x: (dt.datetime.now().date()-x.date()).days, data_to_analyze.stimulus])
    aggregated = blocked.agg({'correct': lambda x: np.mean(x.astype(float))})
    days_passed = np.arange(days)
    stim_number = np.arange(len(stims_all))

    plt.figure()
    plt.subplot(1,2,2)
    cmap = plt.get_cmap('Oranges')
    cmap.set_bad(color = 'k', alpha = 0.5)
    correct = np.zeros((len(days_passed),len(stim_number)),np.float_)

    for day in days_passed:
        for st in stim_number:
            try:
                correct[day,st] = aggregated['correct'][day,str(stims_all[st])]
            except KeyError:
                correct[day,st] = np.nan
    correct = np.ma.masked_invalid(correct)
    plt.pcolormesh(np.rot90(np.fliplr(correct),k=3),cmap=cmap,vmin=0, vmax=1)
    plt.colorbar()
    plt.title(subj)
    plt.xlabel('day')
    plt.ylabel('stim')


def stars(p):
    '''Converts p-values into R-styled stars.

    Signif. codes:
        '***' :  < 0.001
        '**' : < 0.01
        '*' : < 0.05
        '.' : < 0.1
        'n.s.' : < 1.0

    '''
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return 'n.s.'

def plot_stars(p,x,y,size='large',horizontalalignment='center',**kwargs):
    ''' Plots significance stars '''
    plt.text(x,y,stars(p),size=size,horizontalalignment=horizontalalignment,**kwargs)

def plot_linestar(p,x1,x2,y):
    hlines(y, x1, x2)
    plot_stars(0.5*(x1+x2),y+0.02,stars(p),size='large',horizontalalignment='center')
