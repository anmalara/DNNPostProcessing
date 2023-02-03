from tdrstyle import *
import tdrstyle as TDR

TDR.extraText = 'Simulation'

def DataFrameTo2DHist(df, hname, label1,label2, nbinsx=25,nbinsy=50):
    hist = rt.TH2D(hname,hname, nbinsx,df[label1].min(),df[label1].max(), nbinsy,df[label2].min(),df[label2].max())
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

def Plot2DHist(hist, hname, xlabel, ylabel):
    xmin = hist.GetXaxis().GetBinLowEdge(1)
    xmax = hist.GetXaxis().GetBinUpEdge(hist.GetXaxis().GetLast())
    ymin = hist.GetYaxis().GetBinLowEdge(1)
    ymax = hist.GetYaxis().GetBinUpEdge(hist.GetYaxis().GetLast())
    canv,tdrStyle = tdrCanvas(hname, xmin, xmax, ymin, ymax, xlabel, ylabel, square=kSquare, is2D=True, isExtraSpace=True, iPos=0)
    hist.Scale(100./hist.Integral())
    SetAlternative2DColor(hist)
    hist.GetZaxis().SetRangeUser(0,1)
    hist.Draw('same colz')
    canv.SaveAs('PDFs/'+hname+'.pdf')

def PlotCorrelation(hist, hname):
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

def getDataFrame(fname, treename='Events', branches=None):
    import pandas as pd
    if '.root' in fname:
        from root_numpy import root2array, rec2array
        mymatrix = rec2array(root2array(filenames=fname, treename=treename, branches=branches))
        df = pd.DataFrame(mymatrix,columns=[x.replace('m_','').replace('_is_signal','') for x in branches])
    if '.coffea' in fname:
        acc = load(fname)
        d = {}
        for tn in acc.keys():
            if not tn.startswith("tree"): continue
            for region in acc[tn].keys():
                for var in acc[tn][region].keys():
                    if not var in branches: continue
                    d[var] = acc[tn][region][var].value
        df = pd.DataFrame.from_dict(d)
    return df

def filterDataFrame(df, label, val):
    df_ = df[df[label]==val]
    df_.drop([label], axis=1, inplace=True)
    return df_

def RunCorrelation(fname, treename='Events', branches=None, labels_2D=['score', 'mjj']):
    from itertools import combinations
    df = getDataFrame(fname=fname, treename=treename, branches=branches)
    for sig in [0,1]:
        df_ = filterDataFrame(df, 'is_signal', sig)
        corr = df_.corr()
        for label1,label2 in list(combinations(labels_2D, 2)):
            print(label1,label2)
            TDR.cms_lumi= 'corr='+str(int(corr[label1][label2]*100)/100.)
            hname = '2D_'+label1+'_'+label2+'_sig'+str(sig)
            hist = DataFrameTo2DHist(df=df_,hname=hname, label1=label1,label2=label2)
            Plot2DHist(hist, hname, xlabel=label1, ylabel=label2)
            TDR.cms_lumi=''
            hname = 'correlation_sig'+str(sig)
            hist = DataFrameTo2DCorrelationHist(df=corr,hname=hname)
            PlotCorrelation(hist, hname)

        if not 'eventCategory' in df: continue
        for cat in list(set(df['eventCategory'])):
            df_save = df_
            df_ = filterDataFrame(df_, 'eventCategory', cat)
            corr = df_.corr()
            TDR.cms_lumi= 'corr='+str(int(corr['score']['mjj']*100)/100.)
            hname = '2D_score_mjj_sig'+str(sig)+'_cat'+str(cat)
            hist = DataFrameTo2DHist(df=df_,hname=hname, label1='score',label2='mjj')
            Plot2DHist(hist, hname, xlabel='DNN score', ylabel='m(jj)')
            TDR.cms_lumi=''
            hname = 'correlation_sig'+str(sig)+'_cat'+str(cat)
            hist = DataFrameTo2DCorrelationHist(df=corr,hname=hname)
            PlotCorrelation(hist, hname)
            df_ = df_save


def main():
    folder = 'trainings/particlenet_pf/20221216-120722_particlenet_pf_ranger_lr0.001_batch512_VBF_points_features_100_epoch_15_cat012/'
    branches = ['is_signal', 'score_is_signal', 'm_eventCategory', 'm_mjj', 'm_n_PF_jet1', 'm_n_PF_jet2', 'm_PF_UE_charged_size', 'm_PF_UE_neutral_size', 'm_n_nonVBF_jets']
    RunCorrelation(fname=folder+'predict_output/pred_val.root', branches=branches)

    # folder = '/afs/cern.ch/work/a/anmalara/WorkingArea/Hinv/bucoffea/bucoffea/scripts/'
    # branches = ['is_signal', 'score_is_signal', 'leadak4_pt','leadak4_eta','leadak4_phi','leadak4_nef','leadak4_nhf','leadak4_chf','leadak4_cef']
    # RunCorrelation(fname=folder+'vbfhinv_VBF_HToInvisible_M125_TuneCP5_withDipoleRecoil_2017.coffea', branches=branches)

if __name__ == '__main__':
    main()
