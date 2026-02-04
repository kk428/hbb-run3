import ROOT
import os
import json
import argparse

def draw_PFratio_QCDMC_common(args):
    tag = args.tag
    year = args.year

    with open(f"results/{tag}/{year}/setup.json") as f:
        setup = json.load(f)
        cats = setup["categories"]

    plot_dir = f"results/{tag}/{year}/plots/QCD"
    os.makedirs(plot_dir, exist_ok=True)

    for proc in cats:
        if proc == "mucr":
            continue
        nptbins = len(cats[proc]["bins"])-1
        if proc == "vbf":
            nptbins = len(cats[proc]["bins_pt"])-1


        for i in range(nptbins):
            fbb_path = f"results/{tag}/{year}/datacards/testModel_qcdfit_{proc}_bb_{year}.root"
            fcc_path = f"results/{tag}/{year}/datacards/testModel_qcdfit_{proc}_cc_{year}.root"

            fbb = ROOT.TFile.Open(fbb_path)
            fcc = ROOT.TFile.Open(fcc_path)
            if not fbb or fbb.IsZombie():
                print(f"Error opening file: {fbb_path}")
                continue
            if not fcc or fcc.IsZombie():
                print(f"Error opening file: {fcc_path}")
                continue

            wbb = fbb.Get("w")
            wcc = fcc.Get("w")
            if not wbb or not wcc:
                print("Error: workspace not found in one of the files")
                continue

            data_passbb = wbb.data(f"ptbin{i}{proc}pass{year}bb_data_obs")
            data_passcc = wcc.data(f"ptbin{i}{proc}pass{year}cc_data_obs")
            data_fail   = wbb.data(f"ptbin{i}{proc}fail{year}bb_data_obs")

            if not data_passbb or not data_passcc or not data_fail:
                print(f"Error: one of the data sets not found")
                continue

            c1 = ROOT.TCanvas(f"c_{proc}_{i}", f"c_{proc}_{i}", 600, 600)

            var = wbb.var(setup["observable"]["name"])
            frame1 = var.frame(setup["observable"]["nbins"] - 1)

            bin_label = f"ptbin{i}{proc}"
            print(bin_label)
            pdf_pass_bb_name = f"{bin_label}pass{year}bb_qcd"
            pdf_pass_cc_name = f"{bin_label}pass{year}cc_qcd"

            pdf_pass_bb = wbb.pdf(pdf_pass_bb_name)
            pdf_pass_cc = wcc.pdf(pdf_pass_cc_name)
            if pdf_pass_bb:
                pdf_pass_bb.plotOn(frame1, ROOT.RooFit.LineColor(ROOT.kRed))
            else:
                print(f"Warning: pdf {pdf_pass_bb_name} not found")

            if pdf_pass_cc:
                pdf_pass_cc.plotOn(frame1, ROOT.RooFit.LineColor(ROOT.kGreen))
            else:
                print(f"Warning: pdf {pdf_pass_cc_name} not found")

            data_passbb.plotOn(frame1,
                                ROOT.RooFit.Rescale(1.0 / data_passbb.sumEntries()),
                                ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2),
                                ROOT.RooFit.MarkerColor(ROOT.kRed),
                                ROOT.RooFit.MarkerSize(0.5))
            data_passcc.plotOn(frame1,
                                ROOT.RooFit.Rescale(1.0 / data_passcc.sumEntries()),
                                ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2),
                                ROOT.RooFit.MarkerColor(ROOT.kGreen),
                                ROOT.RooFit.MarkerSize(0.5))
            data_fail.plotOn(frame1,
                                ROOT.RooFit.Rescale(1.0 / data_fail.sumEntries()),
                                ROOT.RooFit.LineColor(ROOT.kBlue),
                                ROOT.RooFit.MarkerColor(ROOT.kBlue),
                                ROOT.RooFit.MarkerSize(0.5))

            ROOT.gPad.SetLeftMargin(0.15)

            frame1.SetMaximum(0.1)
            frame1.SetMinimum(0)

            bin_title = cats[proc]["bin_title"]
            title = f"{proc} {bin_title} bin {i+1}"
            if proc == "vbf":
                title = f"{proc} {bin_title} bin 0"

            bin_width = (setup["observable"]["max"] - setup["observable"]["min"]) / setup["observable"]["nbins"]
            frame1.SetTitle(title)
            frame1.SetYTitle(f"Events / {int(bin_width)} GeV")
            frame1.SetXTitle(setup["observable"]["title"])
            frame1.Draw()

            h_dum1 = ROOT.TH1D("h1", "h1", 1, 0, 1)
            h_dum2 = ROOT.TH1D("h2", "h2", 1, 0, 1)
            h_dum3 = ROOT.TH1D("h3", "h3", 1, 0, 1)
            h_dum4 = ROOT.TH1D("h4", "h4", 1, 0, 1)
            h_dum5 = ROOT.TH1D("h5", "h5", 1, 0, 1)

            h_dum1.SetLineColor(ROOT.kGreen)
            h_dum1.SetMarkerColor(ROOT.kGreen)
            h_dum1.SetMarkerStyle(20)

            h_dum2.SetLineColor(ROOT.kBlue)
            h_dum2.SetMarkerColor(ROOT.kBlue)
            h_dum2.SetMarkerStyle(20)

            h_dum3.SetLineColor(ROOT.kRed)
            h_dum3.SetMarkerColor(ROOT.kRed)
            h_dum3.SetMarkerStyle(20)

            h_dum4.SetLineColor(ROOT.kGreen)
            h_dum4.SetMarkerColor(ROOT.kGreen)
            h_dum4.SetLineWidth(3)

            h_dum5.SetLineColor(ROOT.kRed)
            h_dum5.SetMarkerColor(ROOT.kRed)
            h_dum5.SetLineWidth(3)

            leg = ROOT.TLegend(0.5, 0.7, 0.85, 0.85)
            leg.SetBorderSize(0)
            leg.AddEntry(h_dum3, "QCD MC pass bb", "p")
            leg.AddEntry(h_dum1, "QCD MC pass cc", "p")
            leg.AddEntry(h_dum2, "QCD MC fail", "p")
            leg.AddEntry(h_dum5, "QCD MC fit bb", "l")
            leg.AddEntry(h_dum4, "QCD MC fit cc", "l")
            leg.Draw()

            c1.SaveAs(os.path.join(plot_dir, f"{bin_label}.png"))
            c1.SaveAs(os.path.join(plot_dir, f"{bin_label}.pdf"))

            fbb.Close()
            fcc.Close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--year",
        help="year",
        type=str,
        required=True,
        choices=["2022", "2022EE", "2023", "2023BPix", "Run3"],
    )
    parser.add_argument(
        "--tag",
        help="tag",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)
    draw_PFratio_QCDMC_common(args)
