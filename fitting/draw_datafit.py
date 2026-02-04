import os
import ROOT
import argparse

from hbb.common_vars import LUMI

blind = True

def scale_by_bin_width(hist):
    nbins = hist.GetNbinsX()
    for i in range(1, nbins + 1):  # ROOT bins start at 1
        width = hist.GetBinWidth(i)
        if width > 0:
            hist.SetBinContent(i, hist.GetBinContent(i) * width)
            hist.SetBinError(i, hist.GetBinError(i) * width)
    return hist, width

def draw(args, index: int, region: str, cat: str, logscale: bool = True):

    tag = args.tag
    fit = args.fit
    year = args.year

    common_dir = f"results/{tag}/{year}"

    rZbb = 1.0

    year_loop = [year]
    if year == "Run3":
        year_string = f"{(LUMI['2022-2023'] / 1000.0):0.1f}/fb, 22-23"
        year_loop = ["2022", "2022EE", "2023", "2023BPix"]
    else:
        year_string = f"{(LUMI[year] / 1000.0):0.1f}/fb, {year}"



    thisbin = f"pt{index + 1}"
    thisbin_fit = f"ptbin{index}{cat}"

    if cat in ("vbfhi", "vbflo", "vbf"):
        thisbin = f"mjj{index + 1}"
        thisbin_fit = f"ptbin0{cat}"
        name = f"vbf_{region}_{thisbin}_Jetdata_nominal"
    else:
        name = f"{cat}_{region}_{thisbin}_Jetdata_nominal"

    dataf = ROOT.TFile(f"{common_dir}/signalregion.root", "READ")
    if not dataf or dataf.IsZombie():
        raise RuntimeError("Could not open signalregion.root")


    data_obs = dataf.Get(name)
    if not isinstance(data_obs, ROOT.TH1D):
        raise RuntimeError(f"Could not get histogram {name} from data file")
    
    blind_min = data_obs.FindBin(110)
    blind_max = data_obs.FindBin(140)

    data_obs.SetLineColor(ROOT.kBlack)
    data_obs.SetMarkerColor(ROOT.kBlack)
    data_obs.SetMarkerStyle(20)

    if blind: 
        for i in range(blind_min, blind_max):  
            data_obs.SetBinContent(i, 0)
            data_obs.SetBinError(i, 0)

    filename = f"{common_dir}/datacards/testModel_{year}/fitDiagnosticsTest.root"
    out_name_plot = f"{thisbin_fit}{region.replace('_', '')}{year}"

    f = ROOT.TFile(filename, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open {filename}")
    
    VBF = data_obs.Clone("VBF_empty")
    VBF.Reset()
    ggF = data_obs.Clone("ggF_empty")
    ggF.Reset()
    VH = data_obs.Clone("VH_empty")
    VH.Reset()
    bkgHiggs = data_obs.Clone("bkgHiggs_empty")
    bkgHiggs.Reset()
    VV = data_obs.Clone("VV_empty")
    VV.Reset()
    singlet = data_obs.Clone("singlet_empty")
    singlet.Reset()
    ttbar = data_obs.Clone("ttbar_empty")
    ttbar.Reset()
    Zjets = data_obs.Clone("Zjets_empty")
    Zjets.Reset()
    Zjets2 = data_obs.Clone("Zjets2_empty")
    Zjets2.Reset()
    Zjetsbb = data_obs.Clone("Zjetsbb_empty")
    Zjetsbb.Reset()
    Wjets = data_obs.Clone("Wjets_empty")
    Wjets.Reset()
    qcd = data_obs.Clone("qcd_empty")
    qcd.Reset()
    TotalBkg = data_obs.Clone("TotalBkg_empty")
    TotalBkg.Reset()

    for year in year_loop:
    
        name_plot = f"{thisbin_fit}{region.replace('_', '')}{year}"

        histdirname = None
        if fit == "prefit":
            histdirname = f"shapes_prefit/{name_plot}/"
        elif fit == "postfit":
            histdirname = f"shapes_fit_s/{name_plot}/"
        
        tmp_VBF = f.Get(histdirname + "VBF")
        if(tmp_VBF):
            VBF.Add(tmp_VBF)
        tmp = f.Get(histdirname + "ggF")
        if tmp:
            ggF.Add(tmp)
        tmp_wh = f.Get(histdirname + "WH")
        if tmp_wh:
            VH.Add(tmp_wh)
        tmp_zh = f.Get(histdirname + "ZH")
        if tmp_zh:
            VH.Add(tmp_zh)
        tmp_tth = f.Get(histdirname + "ttH")
        if tmp_tth:
            bkgHiggs.Add(tmp_tth)
        tmp_vv = f.Get(histdirname + "VV")
        if tmp_vv:
            VV.Add(tmp_vv)
        tmp_sing = f.Get(histdirname + "singlet")
        if tmp_sing:
            singlet.Add(tmp_sing)
        tmp_tt = f.Get(histdirname + "ttbar")
        if tmp_tt:
            ttbar.Add(tmp_tt)
        tmp_Zjets = f.Get(histdirname + "Zjetsc")
        if tmp_Zjets:
            Zjets.Add(tmp_Zjets)
        tmp_ewkz = f.Get(histdirname + "EWKZc")
        if tmp_ewkz:
            Zjets.Add(tmp_ewkz)
        tmp_Zjets2 = f.Get(histdirname + "Zjetslight")
        if tmp_Zjets2:
            Zjets2.Add(tmp_Zjets2)
        tmp_ewkz2 = f.Get(histdirname + "EWKZlight")
        if tmp_ewkz2:
            Zjets2.Add(tmp_ewkz2)
        tmp_Zjetsbb = f.Get(histdirname + "Zjetsbb")
        if tmp_Zjetsbb:
            Zjetsbb.Add(tmp_Zjetsbb)
        tmp_ewkzb = f.Get(histdirname + "EWKZbb")
        if tmp_ewkzb:
            Zjetsbb.Add(tmp_ewkzb)
        tmp_Wjets = f.Get(histdirname + "Wjets")
        if tmp_Wjets:
            Wjets.Add(tmp_Wjets)
        tmp_ewk = f.Get(histdirname + "EWKW")
        if tmp_ewk:
            Wjets.Add(tmp_ewk)
        tmp_qcd = f.Get(histdirname + "qcd")
        if tmp_qcd:
            qcd.Add(tmp_qcd)
        tmp_tb = f.Get(histdirname + "total_background")
        if tmp_tb:
            TotalBkg.Add(tmp_tb)

    # VBF
    VBF, scale = scale_by_bin_width(VBF)
    VBF.SetLineColor(ROOT.kGreen + 1)
    VBF.SetMarkerColor(ROOT.kGreen + 1)
    VBF.SetLineWidth(3)

    # ggF
    ggF, scale = scale_by_bin_width(ggF)
    ggF.SetLineColor(ROOT.kRed + 1)
    ggF.SetMarkerColor(ROOT.kRed + 1)
    ggF.SetLineStyle(2)
    ggF.SetLineWidth(3)

    # VH
    VH, scale = scale_by_bin_width(VH)
    VH.SetLineColor(ROOT.kBlue + 1)
    VH.SetMarkerColor(ROOT.kBlue + 1)
    VH.SetLineStyle(2)
    VH.SetLineWidth(3)

    # bkg Higgs
    bkgHiggs, scale = scale_by_bin_width(bkgHiggs)
    bkgHiggs.SetLineWidth(1)
    bkgHiggs.SetLineColor(ROOT.kBlack)
    bkgHiggs.SetFillColor(ROOT.kOrange)

    # VV
    VV, scale = scale_by_bin_width(VV)
    VV.SetLineWidth(1)
    VV.SetLineColor(ROOT.kBlack)
    VV.SetFillColor(ROOT.kOrange - 3)

    # single t
    singlet, scale = scale_by_bin_width(singlet)
    singlet.SetLineWidth(1)
    singlet.SetLineColor(ROOT.kBlack)
    singlet.SetFillColor(ROOT.kPink + 6)

    # ttbar
    ttbar, scale = scale_by_bin_width(ttbar)
    ttbar.SetLineWidth(1)
    ttbar.SetLineColor(ROOT.kBlack)
    ttbar.SetFillColor(ROOT.kViolet - 5)

    # Z + jets
    Zjets, scale = scale_by_bin_width(Zjets)
    Zjets.SetLineWidth(1)
    Zjets.SetLineColor(ROOT.kBlack)
    Zjets.SetFillColor(ROOT.kAzure + 8)

    # Z + jets
    Zjets2, scale = scale_by_bin_width(Zjets2)
    Zjets2.SetLineWidth(1)
    Zjets2.SetLineColor(ROOT.kBlack)
    Zjets2.SetFillColor(ROOT.kPink + 8)

    # Z(bb) + jets
    Zjetsbb, scale = scale_by_bin_width(Zjetsbb)
    Zjetsbb.Scale(rZbb)
    Zjetsbb.SetLineWidth(1)
    Zjetsbb.SetLineColor(ROOT.kBlack)
    Zjetsbb.SetFillColor(ROOT.kAzure - 1)

    # W + jets
    Wjets, scale = scale_by_bin_width(Wjets)
    Wjets.SetLineWidth(1)
    Wjets.SetLineColor(ROOT.kBlack)
    Wjets.SetFillColor(ROOT.kGray)

    # QCD
    qcd, scale = scale_by_bin_width(qcd)
    qcd.SetLineWidth(1)
    qcd.SetLineColor(ROOT.kBlack)
    qcd.SetFillColor(ROOT.kWhite)

    # Total background
    TotalBkg, scale = scale_by_bin_width(TotalBkg)
    TotalBkg.SetMarkerColor(ROOT.kRed)
    TotalBkg.SetMarkerSize(0.000000001)
    TotalBkg.SetLineColor(ROOT.kRed)
    TotalBkg.SetFillColor(ROOT.kRed)
    TotalBkg.SetFillStyle(3003)

    max_val = TotalBkg.GetMaximum()
    if data_obs.GetMaximum() > max_val:
        max_val = data_obs.GetMaximum()

    TotalBkg.GetYaxis().SetRangeUser(0.001, 1000 * max_val)
    if not logscale:
        TotalBkg.GetYaxis().SetRangeUser(0, 1.3 * max_val)

    TotalBkg.GetYaxis().SetTitle(f"Events / {int(scale)} GeV")
    TotalBkg.GetXaxis().SetTitle("m_{sd} [GeV]")

    bkg = ROOT.THStack("bkg", "")
    if logscale:
        bkg.Add(bkgHiggs)
        bkg.Add(VV)
        bkg.Add(singlet)
        bkg.Add(ttbar)
        bkg.Add(Zjets)
        bkg.Add(Zjets2)
        bkg.Add(Zjetsbb)
        bkg.Add(Wjets)
        bkg.Add(qcd)
    else:
        bkg.Add(qcd)
        bkg.Add(Wjets)
        bkg.Add(Zjetsbb)
        bkg.Add(Zjets)
        bkg.Add(Zjets2)
        bkg.Add(ttbar)
        bkg.Add(singlet)
        bkg.Add(VV)
        bkg.Add(bkgHiggs)

    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetOptStat(0)

    c = ROOT.TCanvas(name_plot, name_plot, 600, 600)
    pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.33, 1.0, 1.0)
    pad2 = ROOT.TPad("pad2", "pad2", 0.0, 0.0, 1.0, 0.33)

    pad1.SetBottomMargin(1e-5)
    pad1.SetTopMargin(0.1)
    pad1.SetBorderMode(0)
    pad2.SetTopMargin(1e-5)
    pad2.SetBottomMargin(0.3)
    pad2.SetBorderMode(0)

    pad1.SetLeftMargin(0.15)
    pad2.SetLeftMargin(0.15)
    pad1.Draw()
    pad2.Draw()

    textsize1 = 16.0 / (pad1.GetWh() * pad1.GetAbsHNDC())
    textsize2 = 16.0 / (pad2.GetWh() * pad2.GetAbsHNDC())

    TotalBkg.GetYaxis().SetTitleSize(textsize1)
    TotalBkg.GetYaxis().SetLabelSize(textsize1)
    TotalBkg.GetYaxis().SetTitleOffset(2 * pad1.GetAbsHNDC())

    pad1.cd()
    if logscale:
        pad1.SetLogy()

    print("QCD:", qcd.Integral())
    print("Wjets:", Wjets.Integral())
    print("Zjets:", Zjets.Integral())
    print("Zjetsbb:", Zjetsbb.Integral())
    print("ttbar:", ttbar.Integral())
    print("singlet:", singlet.Integral())
    print("VV:", VV.Integral())
    print("bkgHiggs:", bkgHiggs.Integral())

    TotalBkg.Draw("e2")
    bkg.Draw("histsame")
    ggF.Draw("histsame")
    VBF.Draw("histsame")
    VH.Draw("histsame")
    data_obs.Draw("pesame")
    data_obs.Draw("axissame")

    # Legend
    x1, y1 = 0.6, 0.86
    leg = ROOT.TLegend(x1, y1, x1 + 0.3, y1 - 0.32)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetTextSize(textsize1)

    leg.AddEntry(data_obs, "Data", "p")
    leg.AddEntry(TotalBkg, "Bkg. Unc.", "f")
    leg.AddEntry(qcd, "QCD", "f")
    leg.AddEntry(Wjets, "W", "f")
    leg.AddEntry(Zjets, "Z(cc)", "f")
    leg.AddEntry(Zjets2, "Z(light)", "f")
    leg.AddEntry(Zjetsbb, "Z(bb)", "f")
    leg.AddEntry(ttbar, "t#bar{t}", "f")
    leg.AddEntry(singlet, "Single t", "f")
    leg.AddEntry(VV, "VV", "f")
    leg.AddEntry(bkgHiggs, "Bkg. H", "f")
    leg.AddEntry(ggF, "ggF", "l")
    leg.AddEntry(VBF, "VBF", "l")
    leg.AddEntry(VH, "VH", "l")

    leg.Draw()

    l1 = ROOT.TLatex()
    l1.SetNDC()
    l1.SetTextFont(42)
    l1.SetTextSize(textsize1)
    l1.DrawLatex(0.2, 0.82, "#bf{CMS} Preliminary")

    l2 = ROOT.TLatex()
    l2.SetNDC()
    l2.SetTextFont(42)
    l2.SetTextSize(textsize1)
    l2.DrawLatex(0.7, 0.92, year_string)

    text3 = "BB+CC Fail Region" 
    if region == "pass_bb":
        text3 = "BB Pass Region"
    elif region == "pass_cc":
        text3 = "CC Pass Region"
    elif region == "pass":
        text3 = "BB Pass Region"
    l3 = ROOT.TLatex()
    l3.SetNDC()
    l3.SetTextFont(42)
    l3.SetTextSize(textsize1)
    l3.DrawLatex(0.2, 0.77, text3)

    # Category text
    l4 = ROOT.TLatex()
    l4.SetNDC()
    l4.SetTextFont(42)
    l4.SetTextSize(textsize1)
    if cat.lower() == "ggf":
        text2 = f"ggF category p_{{T}} bin {index+1}"
    elif cat.lower() in ("vbfhi", "vbflo", "vbf"):
        text2 = f"VBF category m_{{jj}} bin {index+1}"
    else:
        text2 = f"VH category p_{{T}} bin {index+1}"
    l4.DrawLatex(0.2, 0.72, text2)

    # ratio panel
    pad2.cd()

    TotalBkg_sub = TotalBkg.Clone("TotalBkg_sub")
    TotalBkg_sub.Reset()
    data_obs_sub = data_obs.Clone("data_obs_ratio")
    data_obs_sub.Reset()

    VBF_sub = VBF.Clone("VBF_sub")
    VBF_sub.Reset()
    ggF_sub = ggF.Clone("ggF_sub")
    ggF_sub.Reset()
    VH_sub = VH.Clone("VH_sub")
    VH_sub.Reset()

    nbins = TotalBkg_sub.GetNbinsX()
    for i in range(1, nbins + 1):
        err_data = data_obs.GetBinError(i)
        if err_data != 0:
            TotalBkg_sub.SetBinError(i, TotalBkg.GetBinError(i) / err_data)
            diff = (data_obs.GetBinContent(i) - TotalBkg.GetBinContent(i)) / err_data
            data_obs_sub.SetBinContent(i, diff)
            data_obs_sub.SetBinError(i, 1.0)
            VBF_sub.SetBinContent(i, VBF.GetBinContent(i) / err_data)
            ggF_sub.SetBinContent(i, ggF.GetBinContent(i) / err_data)
            VH_sub.SetBinContent(i, VH.GetBinContent(i) / err_data)
        else:
            TotalBkg_sub.SetBinError(i, 0)
            data_obs_sub.SetBinContent(i, 0)
            data_obs_sub.SetBinError(i, 0)
            VBF_sub.SetBinContent(i, 0)
            ggF_sub.SetBinContent(i, 0)
            VH_sub.SetBinContent(i, 0)

    TotalBkg_sub.GetYaxis().SetTitleSize(textsize2)
    TotalBkg_sub.GetYaxis().SetLabelSize(textsize2)
    TotalBkg_sub.GetXaxis().SetTitleSize(textsize2)
    TotalBkg_sub.GetXaxis().SetLabelSize(textsize2)
    TotalBkg_sub.GetYaxis().SetTitleOffset(2 * pad2.GetAbsHNDC())
    TotalBkg_sub.GetYaxis().SetTitle("(Data - Bkg)/#sigma_{Data}")
    TotalBkg_sub.SetMarkerSize(0)

    if blind:
        for i in range(blind_min, blind_max):
            data_obs.SetBinContent(i, 0)
            data_obs.SetBinError(i, 0)
            TotalBkg_sub.SetBinError(i, 0)
            data_obs_sub.SetBinContent(i, 0)
            data_obs_sub.SetBinError(i, 0)

    min2 = data_obs_sub.GetMinimum()
    max2 = data_obs_sub.GetMaximum()
    if not "pass" in region:
        max2 += 1.0
        min2 -= 1.0
    TotalBkg_sub.GetYaxis().SetRangeUser(1.3 * min2, 1.3 * max2)

    TotalBkg_sub.Draw("e2")
    data_obs_sub.Draw("pesame")

    # Save
    outdir = f"{common_dir}/plots/{fit}"
    os.makedirs(outdir, exist_ok=True)
    outpng = os.path.join(outdir, f"{out_name_plot}.png")
    outpdf = os.path.join(outdir, f"{out_name_plot}.pdf")
    c.SaveAs(outpng)
    c.SaveAs(outpdf)

    f.Close()
    dataf.Close()

if __name__ == "__main__":


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--year",
        help="year",
        type=str,
        required=True,
        choices=["2022", "2022EE", "2023", "2023BPix","Run3"],
    )
    parser.add_argument(
        "--tag",
        help="tag",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--fit",
        help="fit",
        type=str,
        required=True,
        choices=["prefit","postfit"],
    )
    args = parser.parse_args()

    for reg in ["fail", "pass_bb", "pass_cc"]:
        draw(args, index=0, region=reg, cat="vbf", logscale=False)
        # draw(args, index=1, region=reg, cat="vbfhi", logscale=False)
        for cat in ["vh", "ggf"]:
            for i in range(0, 1):
                draw(args, index=i, region=reg, cat=cat, logscale=False)
