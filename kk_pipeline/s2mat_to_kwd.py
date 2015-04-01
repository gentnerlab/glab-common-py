#!/usr/bin/env python
import argparse, os
import h5py as h5
import numpy as np
from string import Template
from shutil import copyfile

# assume spike2 export to mat with the following parameters:
# - aligned starts
# - all chans same length
# - channel names are "Port_N" where N is the 1-indexed 1401 Port number (and, hopefully, electrode site)

KK_PIPELINE_DIR = os.path.dirname(os.path.realpath(__file__))

# for each electrode, we need a list of channel names.
# list indices correspond to indices in the KWD array
CHANMAP = { 
    'A1x16-5mm50': [
        'Port_6',
        'Port_11',
        'Port_3',
        'Port_14',
        'Port_1',
        'Port_16',
        'Port_2',
        'Port_15',
        'Port_5',
        'Port_12',
        'Port_4',
        'Port_13',
        'Port_7',
        'Port_10',
        'Port_8',
        'Port_9',
        ],
        
        
    'N-Form': [
        'Port_1',
        'Port_2',
        'Port_3',
        'Port_4',
        'Port_5',
        'Port_6',
        'Port_7',
        'Port_8',
        'Port_9',
        'Port_10',
        'Port_11',
        'Port_12',
        'Port_13',
        'Port_14',
        'Port_15',
        'Port_16',
        'Port_17',
        'Port_18',
        'Port_19',
        'Port_20',
        'Port_21',
        'Port_22',
        'Port_23',
        'Port_24',
        'Port_25',
        'Port_26',
        'Port_27',
        'Port_28',
        'Port_29',
        'Port_30',
        'Port_31',
        'Port_32',
        ]  
    }

def read_s2mat(mat,site_map):
    with h5.File(mat, 'r') as f_in:
        n_samp = np.inf

        # hack to deal with weird Spike2 export. Try to delete.
        for ch, site in enumerate(site_map):
            length = f_in[site]['length'][0,0]
            if n_samp > length:
                n_samp = length

        shape = (n_samp,len(site_map)) # samples,channels
        data = np.empty(shape,np.int16)

        for ch, site in enumerate(site_map):
            data[:,ch] = f_in[site]['values'][0,:]
        
    return data


def get_args():

    parser = argparse.ArgumentParser(description='Compile Spike2 epoch .mat files into KlustaKwik KWD file.')
    parser.add_argument('mat_list', 
                       help='a text file listing all of the mat files to compile')
    parser.add_argument('probe', default='A1x16-5mm50',
                       help='probe (edit this file to fix mappings)')

    return parser.parse_args()

def get_fs_from_mat(mat,site_map):
    with h5.File(mat, 'r') as f:
        for ii, ch in enumerate(site_map):
            if ii == 0:
                fs =  1 / f[ch]['interval'][0][0]
            else: # make sure all channels have the same sampling rate
                assert fs == 1 / f[ch]['interval'][0][0]
            return fs


def main():
    args = get_args()

    # get experiment info from file structure
    subj, _, pen, site = os.getcwd().split('/')[-4:]
    exp = '_'.join((subj,pen,site))
    kwd = exp + '.raw.kwd'

    params = {
        'probe': args.probe,
        'exp': exp,
        }

    # open KWD file (destination HDF5)
    print 'Opening %s' % kwd
    with h5.File(kwd, 'w') as kwd_f, open(args.mat_list,'r') as mlist_f:
        # for each mat file in the list
        for rec, mat in enumerate(mlist_f):
            mat = mat.strip()
            # read in data from MAT and write to KWD
            print 'Copying %s into Recording/%s' % (mat,rec)
            data = read_s2mat(mat,CHANMAP[args.probe])
            kwd_f.create_dataset('recordings/%i/data' % rec, data=data)

            # grab parameters from first MAT file
            if rec == 0:
                params['fs'] = get_fs_from_mat(mat,CHANMAP[args.probe])
                params['nchan'] = data.shape[1]
            else: # make sure all recordings have the same sampling rate and num chans
                assert params['fs'] == get_fs_from_mat(mat,CHANMAP[args.probe])
                assert params['nchan'] == data.shape[1]

    # copy over the spike template
    copyfile(os.path.join(KK_PIPELINE_DIR,params['probe'],os.getcwd()))

    # read the parameters template
    params_template_in = os.path.join(KK_PIPELINE_DIR,'params.template')
    with open(params_template_in,'r') as src:
        params_template = Template(params_template_in)

    # write the parameters
    with open('params.prm', 'w') as pf:
        pf.write(params_template.substitute(params))

if __name__ == '__main__':
    main()