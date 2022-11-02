import os, glob, argparse, ROOT
from collections import OrderedDict
from plotting_utils import GetROC, PlotGraphs

def GetInfoFromFolder(folder):
    log = os.path.join(folder,'log.log')
    infos = {}
    if not os.path.exists(log):
        return infos
    with open(log, 'r') as f_log:
        for l in f_log.readlines():
            if not any([x in l for x in ['data_config','network_config','log.log']]):
                continue
            info = l.replace(' - (','').replace(')\n','').replace('\'','').split(', ')[1].split('/')
            if 'data_config' in l:
                infos['config'] = info[1].split('.')[0]
            if 'network_config' in l:
                infos['net'] = info[1].split('.')[0].replace('_pf', '').replace('deepak8', 'deepAK8').replace('particlenet', 'PN').replace('VBF_VBF', 'PN')
            if 'log.log' in l:
                infos['info'] = info[2][info[2].find(infos['config'])+len(infos['config'])+1:].replace('epoch_','')
    infos['config'] = infos['config'].split('.')[0].replace('VBF_points_features','').replace('VBF_features','').replace('_','')
    if infos['config'] == '': infos['config'] ='all'
    if not 'cat' in infos['info']: infos['info'] = 'cat'+infos['info']
    infos['leg'] = ' '.join([infos['net'].replace('deepAK8', 'da8').replace('PN', 'PN '), infos['config'].replace('charged', 'cha').replace('neutral', 'neu').replace('UE', 'UE '), infos['info']])
    return infos


def CompareROCS(infos,pdfname,y_true='is_signal',outfile='/predict_output/pred.root',treename='Events', doComparison=False):
    graphs = OrderedDict()
    for info in infos:
        if doComparison and info== infos[0]:
            graph,auc,acc = GetROC(fname=info['fname']+outfile, treename=treename, y_true = y_true, y_score = 'm_mjj')
            graphs[graph] = {'legendtext': 'mjj', 'auc': auc, 'acc': acc}
            if 'style' in info:
                graphs[graph].update(info['style'])
                graphs[graph]['color'] = ROOT.kBlack
                graphs[graph]['lstyle'] = ROOT.kDashed
        graph,auc,acc = GetROC(fname=info['fname']+outfile, treename=treename, y_true = y_true, y_score = 'score_'+y_true)
        graphs[graph] = {'legendtext': info['leg'].lower(), 'auc': auc, 'acc': acc}
        if 'style' in info:
            graphs[graph].update(info['style'])
    PlotGraphs(graphs,pdfname=pdfname)



def main():
    colors = {
        'mlp':     ROOT.kRed+1,
        'deepAK8': ROOT.kOrange+1,
        'PN':      ROOT.kGreen+2,

        'charged': ROOT.kRed+1,
        'neutral': ROOT.kOrange+1,
        'UE':      ROOT.kGreen+2,
        'VBF':     ROOT.kAzure+2,
        'all':     ROOT.kViolet-3,

        'cat0':    ROOT.kRed+1,
        'cat1':    ROOT.kOrange+1,
        'cat2':    ROOT.kAzure+2,
        'cat012':  ROOT.kViolet-3,
        'catm1':   ROOT.kSpring+10,
        'catm2':   ROOT.kTeal+9,
        'catm3':   ROOT.kGreen+3,
        'catm012': ROOT.kGreen+2,
        'catall':  ROOT.kCyan+1,
        }

    style = {
        'cat0': ROOT.kSolid,
        'cat1': ROOT.kDashed,
        'cat2': ROOT.kDotted,
        }

    outputfolder = os.getenv('ANALYSISPATH')+'/PDFs/'
    os.system('mkdir -p '+outputfolder+'vsNet/')
    os.system('mkdir -p '+outputfolder+'vsConfig/')
    os.system('mkdir -p '+outputfolder+'vsCat/')

    folders = glob.glob(os.getenv('ANALYSISPATH')+'/trainings/*/*/')
    folders = list(filter(lambda folder: os.path.exists(os.path.join(folder,'log.log')), folders))
    folders = list(filter(lambda folder: os.path.exists(os.path.join(folder,'predict_output','pred.root')), folders))
    infos = []
    for folder in folders:
        info = GetInfoFromFolder(folder)
        info['fname'] = folder
        info['leg'] = info['leg'].replace('10_','')
        info['info'] = info['info'].replace('10_','')
        infos.append(info)

    nets = ['mlp', 'deepAK8', 'PN']
    cats = ['0','1','2', '012', 'm1','m2','m3', 'm012', 'all']
    configs = ['charged', 'neutral', 'UE', 'VBF', 'all']

    for config in configs:
        for cat in cats:
            to_plot = []
            for net in nets:
                for info in infos:
                    if net in info['net'] and 'cat'+cat == info['info'] and config in info['config']:
                        to_plot.append(info)
            for info in to_plot:
                info['style'] = {'color': colors[info['net']]}
            CompareROCS(infos=to_plot, pdfname=outputfolder+'vsNet/ROCs_'+config+'_cat'+cat, doComparison=True)

    for net in nets:
        for cat in cats:
            to_plot = []
            for config in configs:
                for info in infos:
                    if net in info['net'] and 'cat'+cat == info['info'] and config in info['config']:
                        to_plot.append(info)
            for info in to_plot:
                info['style'] = {'color': colors[info['config']]}
            CompareROCS(infos=to_plot, pdfname=outputfolder+'vsConfig/ROCs_'+net+'_cat'+cat, doComparison=True)

    for net in nets:
        for config in configs:
            to_plot = []
            for cat in cats:
                for info in infos:
                    if net in info['net'] and 'cat'+cat == info['info'] and config in info['config']:
                        to_plot.append(info)
            for info in to_plot:
                info['style'] = {'color': colors[info['info']]}
            CompareROCS(infos=to_plot, pdfname=outputfolder+'vsCat/ROCs_'+net+'_'+config)

if __name__ == '__main__':
    main()
