import os
import numpy as np
import pandas as pd
from array import array
from tdrstyle import *
import tdrstyle as TDR
from plotting_utils import GetScores, GetCorrelation

def plot_scores(fname, mode='loss', pdfname='scores', writeExtraText = True, extraText  = 'Simulation', extraText2 = 'Work in progress', lumi_text=''):
    outputfolder = os.getenv('ANALYSISPATH')+'/PDFs/'
    TDR.writeExtraText = writeExtraText
    TDR.extraText = extraText
    TDR.extraText2 = extraText2
    TDR.cms_lumi_TeV = lumi_text
    hists = GetScores(fname=fname)
    epsilon=0.05
    y_max= -1
    for h in hists.values():
        h.Scale(1./h.Integral())
        y_max = max(y_max,h.GetMaximum())
    for norm in ['norm','log']:
        isLog = norm=='log'
        canv = tdrCanvas(pdfname, 0-epsilon, 1+epsilon, 1.1*1e-4, 2 if isLog else 1.3*y_max, 'Score', 'A.U.', kSquare)
        canv.SetLogy(isLog)
        leg  = tdrLeg(0.7, 0.7, 0.9, 0.9)
        leg2 = tdrLeg(0.5, 0.7, 0.7, 0.9)
        ref_graph = rt.TGraph()
        ref_graph.SetMarkerStyle(rt.kFullCircle)
        ref_line = rt.TLine()
        ref_line.SetLineWidth(2)
        leg2.AddEntry(ref_graph, 'train', 'P')
        leg2.AddEntry(ref_line,  'test', 'l')
        for name, h in hists.items():
            h.SetLineWidth(2)
            color = rt.kAzure+2 if 'sig' in name else rt.kOrange+1
            style = 'P' if 'train' in name else 'hist'
            tdrDraw(h, style, mcolor=color, lcolor=color, fcolor=color, alpha=0.4)
            if 'train' in name: continue
            leg.AddEntry(h, name.replace('sig_test','VBF ').replace('bkg_test','ggH '), 'l')
        fixOverlay()
        canv.SaveAs(outputfolder+pdfname+'_scores_'+norm+'.pdf')
        canv.Close()

def plot_correlations(fname, var=('m_mjj','m_{jj}'), ranges=(50,200,1800), pdfname='correlations', writeExtraText = True, extraText  = 'Simulation', extraText2 = 'Work in progress', lumi_text=''):
    outputfolder = os.getenv('ANALYSISPATH')+'/PDFs/'
    TDR.writeExtraText = writeExtraText
    TDR.extraText = extraText
    TDR.extraText2 = extraText2
    TDR.cms_lumi_TeV = lumi_text
    epsilon=0.05
    y_max= -1
    cuts=['all','0p9','0p8','0p7']
    colors = {
        'all': rt.kRed+1,
        '0p9': rt.kOrange+1,
        '0p8': rt.kAzure+2,
        '0p7': rt.kGreen+2,
    }
    hists2D = GetCorrelation(fname=fname, var=var, ranges=ranges,cuts=cuts)
    for name in ['sig','bkg']:
        h = hists2D[name]
        canv = tdrCanvas(pdfname, 0-epsilon, 1+epsilon, ranges[1], ranges[2], 'Score', var[1], kSquare)
        canv.SetLogz(True)
        SetAlternative2DColor(h)
        h.Draw('same colz')
        fixOverlay()
        canv.SaveAs(outputfolder+pdfname+'_correlation_'+var[0]+'_'+name+'.pdf')
        canv.Close()
    y_max= -1
    for name in ['sig','bkg']:
        hists2D[name+'all'].Scale(1./hists2D[name+'all'].Integral())
        for cut in cuts:
            # hists2D[name+cut].Scale(hists2D[name+'all'].Integral()/hists2D[name+cut].Integral())
            hists2D[name+cut].Scale(1./hists2D[name+cut].Integral())
            y_max = max(y_max,hists2D[name+cut].GetMaximum())
    canv = tdrCanvas(pdfname, ranges[1], ranges[2], 1.1*1e-4, 2, var[1], 'A.U.', kSquare)
    canv.SetLogy(True)
    leg  = tdrLeg(0.7, 0.7, 0.9, 0.9)
    leg2 = tdrLeg(0.5, 0.7, 0.7, 0.9)
    ref_line_sig = rt.TLine()
    ref_line_sig.SetLineWidth(2)
    ref_line_sig.SetLineStyle(rt.kSolid)
    ref_line_bkg = rt.TLine()
    ref_line_bkg.SetLineWidth(2)
    ref_line_bkg.SetLineStyle(rt.kDashed)
    leg2.AddEntry(ref_line_sig, 'VBF', 'l')
    leg2.AddEntry(ref_line_bkg, 'ggH', 'l')
    for name in ['sig','bkg']:
        for cut in cuts:
            h = hists2D[name+cut]
            lstyle = rt.kSolid if 'sig' in name else rt.kDashed
            color = colors[cut]
            hists2D[name+cut].SetLineWidth(2)
            tdrDraw(h, 'hist', mcolor=color, lcolor=color, lstyle=lstyle, fstyle=0, alpha=0.4)
            if 'bkg' in name: continue
            leg.AddEntry(h, 'cut: '+cut.replace('p', '.').replace('all', 'none'), 'l')
    fixOverlay()
    canv.SaveAs(outputfolder+pdfname+'_sculpting_'+var[0]+'.pdf')
    canv.Close()


def main():
    # fname='trainings/particlenet_pf/20221030-020724_particlenet_pf_ranger_lr0.001_batch512_VBF_points_features_epoch_10_all/predict_output/pred.root'
    fname = 'trainings/particlenet_pf/20221111-181843_particlenet_pf_ranger_lr0.001_batch512_VBF_points_features_epoch_40_cat012/predict_output/pred_test.root'
    # plot_scores(fname, pdfname='PN_VBF_epoch_40_all')
    plot_correlations(fname, pdfname='PN_VBF_epoch_40_all', var=('m_mjj','m_{jj}'), ranges=(50,200,1800))
    plot_correlations(fname, pdfname='PN_VBF_epoch_40_all', var=('m_Zeppenfeld','Zeppenfeld'), ranges=(50,0,200))

if __name__ == '__main__':
    main()
