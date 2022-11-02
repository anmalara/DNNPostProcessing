import os
import numpy as np
import pandas as pd
from array import array
from tdrstyle import *
import tdrstyle as TDR

def plot_losses(jsonName, mode='loss', name='history', min_epoch=0, dynamic=True, writeExtraText = True, extraText  = 'Simulation', extraText2 = 'Work in progress', lumi_text=''):
    inputfolder = os.getenv('ANALYSISPATH')+'/runs/'
    outputfolder = os.getenv('ANALYSISPATH')+'/PDFs/'
    TDR.writeExtraText = writeExtraText
    TDR.extraText = extraText
    TDR.extraText2 = extraText2
    TDR.cms_lumi_TeV = lumi_text
    isLoss = mode =='loss'
    history = pd.read_json(inputfolder+jsonName)
    history = history.rename(columns=dict(enumerate(['time','epoch','loss','val_loss'])))
    history = history.filter(items = list(range(min_epoch,len(history))), axis=0)
    history['epoch'] = history['epoch']+1
    history['deltaTime'] = history['time'].map(lambda x: (x-history['time'][0])/60/60)
    if dynamic:
        ranges = np.array(history[mode].to_list()+history['val_'+mode].to_list())
        ranges = (ranges.min()*0.9, ranges.max()*1.1)
    else:
        ranges = (1E-5, 3 if isLoss else 1.05)
    graphs = {}
    modes = [mode,'val_'+mode]
    canv = tdrCanvas(mode, np.min(history['epoch']), np.max(history['epoch'])*1.1, ranges[0], ranges[1], 'Number of training epochs', 'Loss' if isLoss else 'Prediction accuracy', kSquare)
    leg = tdrLeg(0.60, 0.70 if isLoss else 0.20, 0.95, 0.89 if isLoss else 0.39, 0.035, 42, rt.kBlack)
    for m in modes:
        graphs[m] = rt.TGraph(len(history['epoch']), array('d',history['epoch']), array('d',history[m]))
        graphs[m].SetLineWidth(2)
        color = rt.kAzure+2 if m == mode else rt.kOrange+1
        tdrDraw(graphs[m], 'L', mcolor=color, lcolor=color)
        leg.AddEntry(graphs[m], 'Training set' if m == mode else 'Validation set' , 'l')
    canv.SaveAs(outputfolder+name+'.pdf')
    # canv.SetLogy(1)
    # canv.SaveAs(outputfolder+name+'_logy.pdf')
    canv.Close()

    canv = tdrCanvas(mode, 0, np.max(history['deltaTime'])+1, ranges[0], ranges[1], 'hours', 'Loss' if isLoss else 'Prediction accuracy', kSquare)
    leg = tdrLeg(0.60, 0.70 if isLoss else 0.20, 0.95, 0.89 if isLoss else 0.39, 0.035, 42, rt.kBlack)
    for m in modes:
        graphs[m] = rt.TGraph(len(history['deltaTime']), array('d',history['deltaTime']), array('d',history[m]))
        graphs[m].SetLineWidth(2)
        color = rt.kAzure+2 if m == mode else rt.kOrange+1
        tdrDraw(graphs[m], 'L', mcolor=color, lcolor=color)
        leg.AddEntry(graphs[m], 'Training set' if m == mode else 'Validation set' , 'l')
    canv.SaveAs(outputfolder+name+'_h.pdf')
    # canv.SetLogy(1)
    # canv.SaveAs(outputfolder+name+'_h_logy.pdf')
    canv.Close()

def main():
    jsonName='Oct30_02-07-25_b7g47n2618.cern.chparticlenet_pf_VBF_points_features_epoch_10_all/Oct30_02-07-25_b7g47n2618.cern.chparticlenet_pf_VBF_points_features_epoch_10_all.json'
    plot_losses(jsonName, name='PN_all_epoch_10_all')

if __name__ == '__main__':
    main()
