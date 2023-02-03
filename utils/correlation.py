import pandas as pd
from array import array
from root_numpy import root2array, rec2array
from tdrstyle import *
import tdrstyle as TDR

def DataFrameTo2DHist(df, hname, label1,label2, nbinsx=25,xmin=0,xmax=1, nbinsy=50,ymin=200,ymax=1200):
    hist = rt.TH2D(hname,hname, nbinsx,xmin,xmax, nbinsy,ymin,ymax)
    for bin in list(df.index):
        hist.Fill(df[label1][bin],df[label2][bin])
    return hist

def DataFrameTo2DCorrelationHist(df, hname):
    nbins = len(df)
    bins = array('d',list(range(nbins+1)))
    hist = rt.TH2D(hname,hname, nbins, bins, nbins, bins)
    for x, x_label in enumerate(df.keys()):
        hist.GetXaxis().SetBinLabel(x+1,x_label)
        hist.GetYaxis().SetBinLabel(x+1,x_label)
        for y,y_label in enumerate(df.keys()):
            corr_ = df[x_label][y_label]
            hist.SetBinContent(x+1,y+1, round(corr_,2))
            hist.SetBinContent(y+1,x+1, round(corr_,2))
    return hist

def Plot2DHist(hist, hname):
    canv,tdrStyle = tdrCanvas(hname, 0, 1, 200, 1200, 'DNN score', 'm(jj)', square=kSquare, is2D=True, isExtraSpace=True, iPos=0)
    hist.Scale(100./hist.Integral())
    SetAlternative2DColor(hist)
    hist.GetZaxis().SetRangeUser(0,1)
    hist.Draw('same colz')
    canv.SaveAs('PDFs/'+hname+'.pdf')

def PlotCorrelation(hist, hname):
    TDR.extraText='Simulation'
    tdrStyle = SetStyleCorrelation()
    canv = rt.TCanvas(hname,hname)
    SetAlternative2DColor(hist)
    hist.GetZaxis().SetRangeUser(-1,1)
    hist.Draw('colz text')
    CMS_lumi(canv, 0)
    palette = hist.GetListOfFunctions().FindObject('palette')
    palette.SetX1NDC(0.88)
    palette.SetX2NDC(0.92)
    palette.SetY1NDC(tdrStyle.GetPadBottomMargin())
    palette.SetY2NDC(1-tdrStyle.GetPadTopMargin())
    canv.SaveAs('PDFs/'+hname+'.pdf')

def correlation(fname, treename='Events', branches=None):
    mymatrix = rec2array(root2array(filenames=fname, treename=treename, branches=branches))
    df = pd.DataFrame(mymatrix,columns=[x.replace('m_','') for x in branches])
    for cat in list(set(df['eventCategory']))+['all']:
        for sig in [0,1]:
            df_ = df[(df['is_signal']==sig)]
            df_.drop(['is_signal'], axis=1, inplace=True)
            if cat!='all':
                df_ = df_[df_['eventCategory']==cat]
                df_.drop(['eventCategory'], axis=1, inplace=True)
            corr = df_.corr()
            TDR.extraText=' corr='+str(int(corr['score_is_signal']['mjj']*100)/100.)
            hname = '2D_score_mjj_sig'+str(sig)+'_cat'+str(cat)
            hist = DataFrameTo2DHist(df=df_,hname=hname, label1='score_is_signal',label2='mjj')
            Plot2DHist(hist, hname)
            hname = 'correlation_sig'+str(sig)+'_cat'+str(cat)
            hist = DataFrameTo2DCorrelationHist(df=corr,hname=hname)
            PlotCorrelation(hist, hname)


def main():
    folder = 'trainings/particlenet_pf/20221216-120722_particlenet_pf_ranger_lr0.001_batch512_VBF_points_features_100_epoch_15_cat012/'
    branches = ['is_signal', 'score_is_signal', 'm_eventCategory', 'm_mjj', 'm_n_PF_jet1', 'm_n_PF_jet2', 'm_PF_UE_charged_size', 'm_PF_UE_neutral_size', 'm_n_nonVBF_jets']
    correlation(fname=folder+'predict_output/pred_val.root', branches=branches)

if __name__ == '__main__':
    main()
