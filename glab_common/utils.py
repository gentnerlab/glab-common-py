import scipy.io as sio
import numpy as np

def load_mat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
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
