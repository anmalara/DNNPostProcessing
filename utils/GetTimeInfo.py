#!/usr/bin/env python
import os, argparse
from datetime import datetime, timedelta
from collections import OrderedDict
from ClusterSubmission.ClusterSpecificSettings import ClusterSpecificSettings as CSS
from ClusterSubmission.UserSpecificSettings import UserSpecificSettings as USS

def GetInfoFromLog(fname):
    with open(fname, 'r') as f_:
        info = OrderedDict([
            ('tot', None), ('n_epochs', 0), ('avg', 0),
            ('optimalTime', ''), ('config', ''), ('net', ''), ('mode', '')
            ])
        for l in f_.readlines():
            if 'INFO: Epoch ' in l and 'training' in l:
                info['n_epochs'] += 1
                current_date_time = l.split()[0].strip('[')+' '+l.split()[1].strip(']')
                current_date_time = datetime.strptime(current_date_time[:current_date_time.find(',')], '%Y-%m-%d %H:%M:%S')
                if info['tot']==None:
                    info['tot'] = (current_date_time-current_date_time)
                else:
                    info['tot'] += (current_date_time-previous_date_time)
                previous_date_time = current_date_time
            if any([x in l for x in ['data_config','network_config','model_prefix']]):
                info_ = l.replace(' - (','').replace(')\n','').replace('\'','').split(', ')[1].split('/')
                if 'data_config' in l:
                    info['config'] = info_[1].split('.')[0]
                if 'network_config' in l:
                    info['net'] = info_[1].split('.')[0].replace('_pf', '').replace('deepak8', 'deepAK8').replace('particlenet', 'PN').replace('VBF_VBF', 'PN')
                if 'model_prefix' in l:
                    info['mode'] = info_[2][info_[2].find(info['config'])+len(info['config'])+1:]
    if info['tot']==None: return info
    info['avg'] = str(info['tot']/int(info['n_epochs']))
    info['tot'] = str(info['tot'])
    _, info['optimalTime'] = CSS(USS(os.getenv('USER')).Get('cluster')).getTimeInfo(ref_time = str(info['tot']))
    return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='*', default=[])
    args = parser.parse_args()
    for fname in args.files:
        info = GetInfoFromLog(fname)
        print(fname, info.values())

if __name__ == '__main__':
    main()
