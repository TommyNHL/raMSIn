VERSION
## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using ScikitLearn
using StatsPlots
using Plots

## import training set ##
trainDf = CSV.read("G:\\raMSIn\\XGB_Importance2\\df_train24.csv", DataFrame)
trainDEDf = CSV.read("G:\\raMSIn\\XGB_Importance2\\df_train24_log.csv", DataFrame)
trainDEFSDf = CSV.read("G:\\raMSIn\\XGB_Importance2\\df_train24_std.csv", DataFrame)

## import ext val set ##
extDf = CSV.read("G:\\raMSIn\\XGB_Importance2\\df_ext24.csv", DataFrame)
extDEDf = CSV.read("G:\\raMSIn\\XGB_Importance2\\df_ext24_log.csv", DataFrame)
extDEFSDf = CSV.read("G:\\raMSIn\\XGB_Importance2\\df_ext24_std.csv", DataFrame)

## import FNA set ##
fnaDf = CSV.read("G:\\raMSIn\\XGB_Importance2\\df_FNA24.csv", DataFrame)
fnaDEDf = CSV.read("G:\\raMSIn\\XGB_Importance2\\df_FNA24_log.csv", DataFrame)
fnaDEFSDf = CSV.read("G:\\raMSIn\\XGB_Importance2\\df_FNA24_std.csv", DataFrame)

## concate ##
ingestedDf = vcat(trainDf, extDf)
ingestedDEDf = vcat(trainDEDf, extDEDf)
ingestedDEFSDf = vcat(trainDEFSDf, extDEFSDf)


# ==================================================================================================
## assign variables for TP and TN data ##
trainDf_0 = trainDf[trainDf.type .== 0, :]
trainDf_1 = trainDf[trainDf.type .== 1, :]
extDf_0 = extDf[extDf.type .== 0, :]
extDf_1 = extDf[extDf.type .== 1, :]
fnaDf_0 = fnaDf[fnaDf.type .== 0, :]
fnaDf_1 = fnaDf[fnaDf.type .== 1, :]

trainDEDf_0 = trainDEDf[trainDEDf.type .== 0, :]
trainDEDf_1 = trainDEDf[trainDEDf.type .== 1, :]
extDEDf_0 = extDEDf[extDEDf.type .== 0, :]
extDEDf_1 = extDEDf[extDEDf.type .== 1, :]
fnaDEDf_0 = fnaDEDf[fnaDEDf.type .== 0, :]
fnaDEDf_1 = fnaDEDf[fnaDEDf.type .== 1, :]

trainDEFSDf_0 = trainDEFSDf[trainDEFSDf.type .== 0, :]
trainDEFSDf_1 = trainDEFSDf[trainDEFSDf.type .== 1, :]
extDEFSDf_0 = extDEFSDf[extDEFSDf.type .== 0, :]
extDEFSDf_1 = extDEFSDf[extDEFSDf.type .== 1, :]
fnaDEFSDf_0 = fnaDEFSDf[fnaDEFSDf.type .== 0, :]
fnaDEFSDf_1 = fnaDEFSDf[fnaDEFSDf.type .== 1, :]

ingestedDEFSDf_0 = ingestedDEFSDf[ingestedDEFSDf.type .== 0, :]
ingestedDEFSDf_1 = ingestedDEFSDf[ingestedDEFSDf.type .== 1, :]
ingestedDEDf_0 = ingestedDEDf[ingestedDEDf.type .== 0, :]
ingestedDEDf_1 = ingestedDEDf[ingestedDEDf.type .== 1, :]
ingestedDf_0 = ingestedDf[ingestedDf.type .== 0, :]
ingestedDf_1 = ingestedDf[ingestedDf.type .== 1, :]


# ==================================================================================================
## plot graph ##
using DataSci4Chem
#
layout = @layout [a{0.33w,0.33h} b{0.33w,0.33h} c{0.33w,0.33h} 
                  d{0.33w,0.33h} e{0.33w,0.33h} f{0.33w,0.33h} 
                  g{0.33w,0.33h} h{0.33w,0.33h} i{0.33w,0.33h}]
default(grid = false, legend = false)
gr()
#
outplotTPTNdetaRiDistrution = plot(layout = layout, #link = :both, 
        size = (1800, 1500), margin = (8, :mm), dpi = 300)
#
histogram!(trainDf_0[:, 14], bins = 150, 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Absolute MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Age-Matched Control", 
    fc = "peachpuff4", 
    lc = "peachpuff4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Training Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(trainDf_1[:, 14], bins = 150, 
    subplot = 1, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Absolute MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Hepatocellular Carcinoma", 
    fc = "brown4", 
    lc = "brown4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Training Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(ingestedDf_0[:, 14], bins = 150, 
    subplot = 4, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Age-Matched Control", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Age-Matched Control", 
    fc = "peachpuff4", 
    lc = "peachpuff4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Ingested Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(ingestedDf_1[:, 14], bins = 150, 
    subplot = 4, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Absolute MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Hepatocellular Carcinoma", 
    fc = "brown4", 
    lc = "brown4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Ingested Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(fnaDf_0[:, 14], bins = 150, 
    subplot = 7, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Absolute MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Age-Matched Control", 
    fc = "peachpuff4", 
    lc = "peachpuff4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Application Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(fnaDf_1[:, 14], bins = 150, 
    subplot = 7, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Absolute MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Hepatocellular Carcinoma", 
    fc = "brown4", 
    lc = "brown4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Application Dataset", 
    titlefont = font(12), 
    dpi = 300)


histogram!(trainDEDf_0[:, 14], bins = 150, 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Relative MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Age-Matched Control", 
    fc = "peachpuff4", 
    lc = "peachpuff4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Scaled\nTraining Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(trainDEDf_1[:, 14], bins = 150, 
    subplot = 2, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Relative MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Hepatocellular Carcinoma", 
    fc = "brown4", 
    lc = "brown4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Scaled\nTraining Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(ingestedDEDf_0[:, 14], bins = 150, 
    subplot = 5, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Relative MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Age-Matched Control", 
    fc = "peachpuff4", 
    lc = "peachpuff4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Scaled\nIngested Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(ingestedDEDf_1[:, 14], bins = 150, 
    subplot = 5, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Relative MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Hepatocellular Carcinoma", 
    fc = "brown4", 
    lc = "brown4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Scaled\nIngested Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(fnaDEDf_0[:, 14], bins = 150, 
    subplot = 8, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Relative MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Age-Matched Control", 
    fc = "peachpuff4", 
    lc = "peachpuff4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Scaled\nApplication Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(fnaDEDf_1[:, 14], bins = 150, 
    subplot = 8, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Relative MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Hepatocellular Carcinoma", 
    fc = "brown4", 
    lc = "brown4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Scaled\nApplication Dataset", 
    titlefont = font(12), 
    dpi = 300)


histogram!(trainDEFSDf_0[:, 14], bins = 150, 
    subplot = 3, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "z-score of Relative MS Signal Intensity)", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Age-Matched Control", 
    fc = "peachpuff4", 
    lc = "peachpuff4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Standardized\nTraining Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(trainDEFSDf_1[:, 14], bins = 150, 
    subplot = 3, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "z-score of relative MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Hepatocellular Carcinoma", 
    fc = "brown4", 
    lc = "brown4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Standardized\nTraining Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(ingestedDEFSDf_0[:, 14], bins = 150, 
    subplot = 6, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "z-score of Relative MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Age-Matched Control", 
    fc = "peachpuff4", 
    lc = "peachpuff4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Standardized\nIngested Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(ingestedDEFSDf_1[:, 14], bins = 150, 
    subplot = 6, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "z-score of Relative MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Hepatocellular Carcinoma", 
    fc = "brown4", 
    lc = "brown4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Standardized\nIngested Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(fnaDEFSDf_0[:, 14], bins = 150, 
    subplot = 9, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "z-score of Relative MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Age-Matched Control", 
    fc = "peachpuff4", 
    lc = "peachpuff4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Standardized\nApplication Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(fnaDEFSDf_1[:, 14], bins = 150, 
    subplot = 9, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "z-score of Relative MS Signal Intensity", xguidefontsize=10, 
    ylabel = "Count", yguidefontsize=10, 
    label = "Hepatocellular Carcinoma", 
    fc = "brown4", 
    lc = "brown4", 
    lw = 1, 
    margin = (5, :mm), 
    xtickfontsize = 8, 
    ytickfontsize= 8, 
    legend = :topright, 
    legendfont = font(8), 
    title = "Standardized\nApplication Dataset", 
    titlefont = font(12), 
    dpi = 300)

## save ##
savefig(outplotTPTNdetaRiDistrution, "G:\\raMSIn\\XGB_Importance2\\outplot_type0or1Distrution.png")
