import numpy as np
import pandas as pd
from array import array
from collections import OrderedDict
from root_numpy import root2array, rec2array
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils import check_consistent_length, assert_all_finite, column_or_1d, check_array
from tdrstyle import *
import tdrstyle as TDR

def binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]

def roc_curve_reduced(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True):
    # Copied from https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/metrics/ranking.py#L535
    # Extended by purity-part

    fps, tps, thresholds = binary_clf_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    if drop_intermediate and len(fps) > 2:
        keep_only_every = 1 if (len(fps) < 10000) else (10 if (len(fps) < 100000) else 100)
        optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]
        reduced_idxs = range(len(thresholds))[0::keep_only_every]
        if not len(thresholds)-1 == reduced_idxs[-1]:
            reduced_idxs.append(len(thresholds)-1)
        fps = fps[reduced_idxs]
        tps = tps[reduced_idxs]
        thresholds = thresholds[reduced_idxs]

    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return (fpr, tpr, thresholds)

def list_to_tgraph(x, y):
    if not type(x) == type(y):
        raise ValueError('In \'list_to_tgraph(): Passed two objects of different type.\'')
    if not len(x) == len(y):
        raise ValueError('In \'list_to_tgraph(): Passed two lists with different length.\'')
    x = array('f', x)
    y = array('f', y)
    g = rt.TGraph(len(x), x, y)
    return g

def PlotGraphs(graphs={}, pdfname='ROCs', x_title='Signal efficiency', y_title='Background efficiency', x_range=(-0.1, 1.1), y_range=(1e-04, 1.2), logy=True, writeExtraText = True, extraText  = 'Simulation', extraText2 = 'Work in progress', lumi_text=''):
    TDR.writeExtraText = writeExtraText
    TDR.extraText = extraText
    TDR.extraText2 = extraText2
    TDR.cms_lumi_TeV = lumi_text

    canv = tdrCanvas('graphs', x_range[0], x_range[1], y_range[0], y_range[1], x_title, y_title, kSquare)

    if logy:
        # canv = tdrCanvas('ROCs', -0.1, 1.1, 1e-04, 1.2, x_title, y_title, kSquare)
        # leg = tdrLeg(0.40, 0.2, 0.89, 0.35, 0.03, 42, rt.kBlack)
        leg = tdrLeg(0.30, 0.15, 0.89, 0.15+0.03*(len(graphs)+1), 0.03, 42, rt.kBlack)
    else:
        # canv = tdrCanvas('ROCs', -0.1, 1.1, 0, 1.5, x_title, y_title, kSquare)
        leg = tdrLeg(0.30, 0.65, 0.95, 0.9, 0.035, 42, rt.kBlack)
    canv.SetLogy(logy)
    for graph, info in graphs.items():
        color = info['color'] if 'color' in info else rt.kRed
        lstyle = info['lstyle'] if 'lstyle' in info else rt.kSolid
        graph.SetLineWidth(2)
        tdrDraw(graph, 'L', mcolor=color, lcolor=color, lstyle=lstyle)
        legstring = info['legendtext']
        if 'auc' in info: legstring += ', AUC: %1.3f'%(info['auc'])
        if 'acc' in info: legstring += ', acc: %1.3f'%(info['acc'])
        leg.AddEntry(graph, legstring, 'l')
    canv.SaveAs(pdfname+'.pdf')
    canv.Close()


def GetDNNOutputInfos(df, y_true = 'y_true', y_score = 'y_score', swap=False, drop_intermediate=True):
    acc = accuracy_score(df[y_true].round().astype(int),df[y_score].round().astype(int)) if len(df[y_score][df[y_score]>1])==0 else 0
    fpr, tpr, thr = roc_curve_reduced(df[y_true], df[y_score]) if drop_intermediate else roc_curve(df[y_true], df[y_score])
    if swap: fpr, tpr = (tpr,fpr)
    auc_ = auc(fpr, tpr)
    return (tpr,fpr,auc_,acc)


def GetROC(fname='mlp_predict.root', treename='Events', y_true = 'is_signal', y_score = 'score_is_signal', swap=False, drop_intermediate=True):
    mymatrix = rec2array(root2array(filenames=fname, treename=treename, branches=[y_true,y_score]))
    df = pd.DataFrame(mymatrix,columns=['y_true','y_score'])
    tpr,fpr,auc_,acc = GetDNNOutputInfos(df, y_true = 'y_true', y_score = 'y_score', swap=swap, drop_intermediate=drop_intermediate)
    graph = list_to_tgraph(tpr,fpr)
    return (graph,auc_,acc)


def GetROCvsCat(fname='mlp_predict.root', treename='Events', y_true = 'is_signal', y_score = 'score_is_signal', catVar='m_eventCategory', categories={}, swap=False, drop_intermediate=True):
    mymatrix = rec2array(root2array(filenames=fname, treename=treename, branches=[y_true,y_score,catVar,'m_mjj']))
    df = pd.DataFrame(mymatrix,columns=['y_true','y_score',catVar,'mjj'])
    graphs = OrderedDict()
    for cat,color in categories.items():
        df_ = df[df[catVar]==int(cat)]
        tpr,fpr,auc_,acc = GetDNNOutputInfos(df_, y_true = 'y_true', y_score = 'y_score', swap=swap, drop_intermediate=drop_intermediate)
        graph = list_to_tgraph(tpr,fpr)
        graphs[graph] = {'legendtext': 'PN Cat. '+str(int(cat)), 'auc': auc_, 'acc': acc, 'color':color, 'lstyle':rt.kSolid}
        tpr,fpr,auc_,acc = GetDNNOutputInfos(df_, y_true = 'y_true', y_score = 'mjj', swap=swap, drop_intermediate=drop_intermediate)
        graph = list_to_tgraph(tpr,fpr)
        graphs[graph] = {'legendtext': 'm_{jj}  Cat. '+str(int(cat)), 'auc': auc_, 'color':color, 'lstyle':rt.kDashed}
    return graphs


def GetScores(fname='mlp_predict.root', treename='Events', y_true = 'is_signal', y_score = 'score_is_signal', compareTrain=True):
    mymatrix = rec2array(root2array(filenames=fname, treename=treename, branches=[y_true,y_score]))
    mymatrix_train = rec2array(root2array(filenames=fname.replace('test','train'), treename=treename, branches=[y_true,y_score]))
    dfs = {}
    dfs['test'] = pd.DataFrame(mymatrix,columns=['y_true','y_score'])
    dfs['train'] = pd.DataFrame(mymatrix_train,columns=['y_true','y_score'])
    hists = {}
    for name in ['sig','bkg']:
        for mode in ['test','train']:
            if not compareTrain and mode !='test': continue
            hname = name+'_'+mode
            hists[hname] = rt.TH1F('h_'+hname, 'h_'+hname, 50, 0,1)
            ref = 1 if 'sig' in name else 0
            df = dfs[mode]
            var = array('d', df[df['y_true']==ref]['y_score'])
            for x in var: hists[hname].Fill(x)
    return hists


def GetCorrelation(fname='mlp_predict.root', treename='Events', y_true = 'is_signal', y_score = 'score_is_signal', var=('is_signal','y_true'), ranges=(100,0,1), cuts=['all','0p9','0p8','0p7']):
    mymatrix = rec2array(root2array(filenames=fname, treename=treename, branches=[y_true,y_score,var[0]]))
    df = pd.DataFrame(mymatrix,columns=['y_true','y_score',var[1]])
    hists = {}
    for hname in ['sig','bkg']:
        hists[hname] = rt.TH2F('h_'+hname, 'h_'+hname, 20, 0,1, ranges[0],ranges[1],ranges[2])
        for cut in cuts:
            hists[hname+cut] = rt.TH1F('h_'+hname+cut, 'h_'+hname+cut, ranges[0],ranges[1],ranges[2])
        ref = 1 if 'sig' in hname else 0
        mask = df['y_true']==ref
        var_x = array('d', df[mask]['y_score'])
        var_y = array('d', df[mask][var[1]])
        for i in range(len(var_x)):
            hists[hname].Fill(var_x[i],var_y[i])
            for cut in cuts:
                if cut!='all' and var_x[i]<float(cut.replace('p','.')): continue
                hists[hname+cut].Fill(var_y[i])
    return hists
