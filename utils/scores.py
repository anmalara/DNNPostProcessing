import os
import numpy as np
import pandas as pd
from array import array
from tdrstyle import *
import tdrstyle as TDR
from plotting_utils import GetScores

def plot_scores(fname, mode='loss', pdfname='scores', min_epoch=0, dynamic=True, writeExtraText = True, extraText  = 'Simulation', extraText2 = 'Work in progress', lumi_text=''):
    outputfolder = os.getenv('ANALYSISPATH')+'/PDFs/'
    TDR.writeExtraText = writeExtraText
    TDR.extraText = extraText
    TDR.extraText2 = extraText2
    TDR.cms_lumi_TeV = lumi_text
    hists = {}
    hists['VBF'], hists['ggH'] = GetScores(fname=fname)
    epsilon=0.05
    y_max= -1
    for name,h in hists.items():
        h.Scale(1./h.Integral())
        y_max = max(y_max,h.GetMaximum())
    for norm in ['norm','log']:
        isLog = norm=='log'
        canv = tdrCanvas(pdfname, 0-epsilon, 1+epsilon, 1.1*1e-4, 2 if isLog else 1.3*y_max, 'Score', 'A.U.', kSquare)
        canv.SetLogy(isLog)
        leg = tdrLeg(0.7, 0.7, 0.9, 0.9)
        for name in ['ggH','VBF']:
            h = hists[name]
            h.SetLineWidth(2)
            color = rt.kAzure+2 if name =='VBF' else rt.kOrange+1
            tdrDraw(h, 'hist', mcolor=color, lcolor=color, fcolor=color, alpha=0.4)
            leg.AddEntry(h, name , 'l')
        fixOverlay()
        canv.SaveAs(outputfolder+pdfname+'_scores_'+norm+'.pdf')
        canv.Close()

def main():
    fname='trainings/particlenet_pf/20221030-020724_particlenet_pf_ranger_lr0.001_batch512_VBF_points_features_epoch_10_all/predict_output/pred.root'
    plot_scores(fname, pdfname='PN_all_epoch_10_all')

if __name__ == '__main__':
    main()
