#! /usr/bin/env python
import os, json, glob
from datetime import datetime
from collections import OrderedDict
from printing_utils import green
from ClusterSubmission.CondorBase import SubmitListToCondor, ResubmitFromJson
from utils.GetTimeInfo import GetInfoFromLog

def GetArgs(args):
    empty = list(filter(lambda x: args[x]=='', args.keys()))
    if len(empty):
        raise ValueError('Some arguments are empty:'+ str(empty))
    return ' '.join(args.values())



def submit(n_epochs, cat, doTesting, resubmit, debug):
    JsonInfo = {
        'should_transfer_files': 'YES',
        'transfer_output_files': 'DNNPostProcessing/output.tar',
        'transfer_output_remaps': '"output.tar = $(outdir)/output.$(ClusterId).$(ProcId).tar"',
        'request_GPUs': '1',
        'request_CPUs': '4',
        'getenv': 'False',
        'MY.SendCredential': 'True',
        'request_memory': '4',
        'request_disk': '4',
        'on_exit_remove':'(ExitBySignal == False) && (ExitCode == 0)',
        'max_retries': '3',
        'requirements': 'Machine =!= LastRemoteHost',
    }
    deleteInfo = [
        #'request_disk', 'request_memory'
        ]

    times_cat = {
        'cat0':    {'cat': 'eventCategory_[0]',    'time':'00:05:00',},
        'cat1':    {'cat': 'eventCategory_[1]',    'time':'00:05:00',},
        'cat2':    {'cat': 'eventCategory_[2]',    'time':'00:05:00',},
        'catm1':   {'cat': 'eventCategory_-[1]',   'time':'00:30:00',},
        'catm2':   {'cat': 'eventCategory_-[2]',   'time':'00:30:00',},
        'catm3':   {'cat': 'eventCategory_-[3]',   'time':'00:30:00',},
        'cat012':  {'cat': 'eventCategory_[0-2]',  'time':'00:30:00',},
        'catm123': {'cat': 'eventCategory_-[0-2]', 'time':'00:30:00',},
        'all':     {'cat': 'all',                  'time':'02:00:00',},
    }

    model_config_info = []
    modes = ['', '_charged', '_neutral', '_UE', '_VBF']
    modes = ['']
    modes = ['_70','_80','_90','_100','_110','_120','_sorted_70','_sorted_80','_sorted_90','_sorted_100','_sorted_110','_sorted_120']
    for mode in modes:
        # model_config_info.append(('mlp_pf',         'VBF_features'+mode))
        # model_config_info.append(('deepak8_pf',     'VBF_features'+mode))
        model_config_info.append(('particlenet_pf', 'VBF_points_features'+mode))

    job_args   = []
    inputdir ='/eos/home-a/anmalara/Public/DNNInputs/'
    args = OrderedDict([
        # ('filepath',   'DNNPostProcessing'),
        ('filepath',   os.getcwd()),
        ('model',      ''),
        ('data',       ''),
        ('flag_test',  'none'),
        ('train',      inputdir+'/'+times_cat[cat]['cat']+'/MC*M1[2][4-6]*UL1[6-7-8]*.root'),
        ('val',        inputdir+'/'+times_cat[cat]['cat']+'/MC*M1[2][0]*UL1[6-7-8]*.root'),
        ('test',       inputdir+'/'+times_cat[cat]['cat']+'/MC*M1[3][0]*UL1[6-7-8]*.root'),
        ('n_gpus',     '0' if not 'request_GPUs' in JsonInfo else JsonInfo['request_GPUs']),
        ('n_epochs',   n_epochs),
        ('extra_name', 'epoch_'+n_epochs+'_'+cat),
    ])
    if doTesting:
        args['flag_test'], args['train'], args['val'], args['test'] = ('test', 'none','none','none')

    if args['flag_test']!= 'test' and any([ args[x]== 'none' for x in ['train','val','test']]):
        raise ValueError('Unexpected inputs.')

    for model,data in model_config_info:
        args['model'] = model
        args['data'] = data
        job_args.append(GetArgs(args))

    outdir     = os.getenv('HOME')+'/workspace/CondorOutputs/'
    executable = 'run_on_condor.sh'
    Time = str((datetime.strptime(times_cat[cat]['time'], '%H:%M:%S') - datetime(1900, 1, 1))*int(n_epochs)).replace(' days, ','-')
    ClusterId_info_name = outdir+'ClusterId_info.json'
    ClusterId_info = {}
    ClusterIds = []
    if os.path.exists(ClusterId_info_name):
        with open(ClusterId_info_name, 'r') as f:
            ClusterId_info = json.load(f)
            if args['extra_name'] in ClusterId_info: ClusterIds = ClusterId_info[args['extra_name']]
    if resubmit:
        jsonName = outdir+'JobInfo_'+args['extra_name']
        with open(jsonName+'.json', 'r') as f:
            ClusterIds.append(json.load(f)['ClusterId'])
        ClusterIds = list(set(ClusterIds)-set([-1]))
        to_remove = []
        for ClusterId in ClusterIds:
            for fname in glob.glob(outdir+'*'+ClusterId+'*out'):
                info = GetInfoFromLog(fname=fname)
                if os.path.exists(fname.replace('run_on_condor_','output.').replace('.out','.tar').replace('_','.')):
                    to_remove.append(' '.join([info['net'].replace('deepAK8', 'deepak8').replace('PN','particlenet')+'_pf',info['config'],'']))
        ClusterId = ResubmitFromJson(jsonName=jsonName, to_remove=to_remove, debug=debug)
    else:
        ClusterId = SubmitListToCondor(job_args, executable=executable, outdir=outdir, Time=Time, JsonInfo=JsonInfo, deleteInfo=deleteInfo, jsonName=args['extra_name'], debug=debug)
    ClusterIds.append(ClusterId)
    ClusterId_info[args['extra_name']] = list(set(ClusterIds)-set([-1]))
    if len(ClusterId_info[args['extra_name']])==0:
        del ClusterId_info[args['extra_name']]
    if len(ClusterId_info)!=0:
        with open(ClusterId_info_name, 'w') as f:
            json.dump(ClusterId_info, f, sort_keys=True, indent=4)


def main():
    epochs = ['20']
    # categories = ['cat0','cat1','cat2','catm1','catm2','catm3','cat012','catm123','all']
    #categories = ['catm2','catm3','cat012','catm123']
    # categories = ['all']
    categories = ['cat012']

    doTesting = False
    # doTesting = True

    debug=False
    # debug=True

    resubmit=True
    resubmit=False
    for n_epochs in epochs:
        for cat in categories:
            print(green('--> Working on: n_epochs:'+n_epochs+' '+cat ))
            submit(n_epochs=n_epochs, cat=cat, doTesting=doTesting, resubmit=resubmit, debug=debug)

if __name__ == '__main__':
    main()
