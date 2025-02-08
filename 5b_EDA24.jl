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
trainDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_train_raMSIn4nonInDI.csv", DataFrame)
trainDEDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_train_raMSIn4nonInDI_log.csv", DataFrame)
trainDEFSDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_train_raMSIn4nonInDI_norm.csv", DataFrame)

## import ext val set ##
extDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_ext_raMSIn4nonInDI.csv", DataFrame)
extDEDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_ext_raMSIn4nonInDI_log.csv", DataFrame)
extDEFSDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_ext_raMSIn4nonInDI_norm.csv", DataFrame)

## import ingested set ##
ingestedDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_ingested_raMSIn4nonInDI.csv", DataFrame)
ingestedDEDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_ingested_raMSIn4nonInDI_log.csv", DataFrame)
ingestedDEFSDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_ingested_raMSIn4nonInDI_norm.csv", DataFrame)

## import FNA set ##
fnaDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_FNA_raMSIn4nonInDI.csv", DataFrame)
fnaDEDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_FNA_raMSIn4nonInDI_log.csv", DataFrame)
fnaDEFSDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_FNA_raMSIn4nonInDI_norm.csv", DataFrame)

## import DirectIn set ##
diDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_nonInDI_raMSIn4nonInDI.csv", DataFrame)
diDEDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_nonInDI_raMSIn4nonInDI_log.csv", DataFrame)
diDEFSDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\df_nonInDI_raMSIn4nonInDI_norm.csv", DataFrame)


# ==================================================================================================
## assign variables for TP and TN data ##
trainDf_0 = trainDf[trainDf.type .== 0, :]
trainDf_1 = trainDf[trainDf.type .== 1, :]
extDf_0 = extDf[extDf.type .== 0, :]
extDf_1 = extDf[extDf.type .== 1, :]
ingestedDf_0 = ingestedDf[ingestedDf.type .== 0, :]
ingestedDf_1 = ingestedDf[ingestedDf.type .== 1, :]
fnaDf_0 = fnaDf[fnaDf.type .== 0, :]
fnaDf_1 = fnaDf[fnaDf.type .== 1, :]
diDf_0 = diDf[diDf.type .== 0, :]
diDf_1 = diDf[diDf.type .== 1, :]

trainDEDf_0 = trainDEDf[trainDEDf.type .== 0, :]
trainDEDf_1 = trainDEDf[trainDEDf.type .== 1, :]
extDEDf_0 = extDEDf[extDEDf.type .== 0, :]
extDEDf_1 = extDEDf[extDEDf.type .== 1, :]
ingestedDEDf_0 = ingestedDEDf[ingestedDEDf.type .== 0, :]
ingestedDEDf_1 = ingestedDEDf[ingestedDEDf.type .== 1, :]
fnaDEDf_0 = fnaDEDf[fnaDEDf.type .== 0, :]
fnaDEDf_1 = fnaDEDf[fnaDEDf.type .== 1, :]
diDEDf_0 = diDEDf[diDEDf.type .== 0, :]
diDEDf_1 = diDEDf[diDEDf.type .== 1, :]

trainDEFSDf_0 = trainDEFSDf[trainDEFSDf.type .== 0, :]
trainDEFSDf_1 = trainDEFSDf[trainDEFSDf.type .== 1, :]
extDEFSDf_0 = extDEFSDf[extDEFSDf.type .== 0, :]
extDEFSDf_1 = extDEFSDf[extDEFSDf.type .== 1, :]
ingestedDEFSDf_0 = ingestedDEFSDf[ingestedDEFSDf.type .== 0, :]
ingestedDEFSDf_1 = ingestedDEFSDf[ingestedDEFSDf.type .== 1, :]
fnaDEFSDf_0 = fnaDEFSDf[fnaDEFSDf.type .== 0, :]
fnaDEFSDf_1 = fnaDEFSDf[fnaDEFSDf.type .== 1, :]
diDEFSDf_0 = diDEFSDf[diDEFSDf.type .== 0, :]
diDEFSDf_1 = diDEFSDf[diDEFSDf.type .== 1, :]


# ==================================================================================================
## plot graph ##
using DataSci4Chem
#
layout = @layout [a{0.33w,0.2h} b{0.33w,0.2h} c{0.33w,0.2h} 
                  d{0.33w,0.2h} e{0.33w,0.2h} f{0.33w,0.2h} 
                  g{0.33w,0.2h} h{0.33w,0.2h} i{0.33w,0.2h}
                  j{0.33w,0.2h} k{0.33w,0.2h} l{0.33w,0.2h} 
                  m{0.33w,0.2h} n{0.33w,0.2h} o{0.33w,0.2h}]
default(grid = false, legend = false)
gr()
#
outplotTPTNdetaRiDistrution = plot(layout = layout, #link = :both, 
        size = (1600, 1500), margin = (8, :mm), dpi = 300)
#
histogram!(trainDf_1[:, 8], bins = 150, 
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
    histogram!(trainDf_0[:, 8], bins = 150, 
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
histogram!(extDf_1[:, 8], bins = 150, 
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
    title = "External Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(extDf_0[:, 8], bins = 150, 
    subplot = 4, 
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
    title = "External Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(ingestedDf_1[:, 8], bins = 150, 
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
    title = "Ingested Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(ingestedDf_0[:, 8], bins = 150, 
    subplot = 7, 
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
histogram!(fnaDf_1[:, 8], bins = 150, 
    subplot = 10, 
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
    title = "Smear MSI Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(fnaDf_0[:, 8], bins = 150, 
    subplot = 10, 
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
    title = "Smear MSI Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(diDf_1[:, 8], bins = 150, 
    subplot = 13, 
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
    title = "Direct Infusion Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(diDf_0[:, 8], bins = 150, 
    subplot = 13, 
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
    title = "Direct Infusion Dataset", 
    titlefont = font(12), 
    dpi = 300)


histogram!(trainDEDf_1[:, 8], bins = 150, 
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
    histogram!(trainDEDf_0[:, 8], bins = 150, 
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
histogram!(extDEDf_1[:, 8], bins = 150, 
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
    title = "Scaled\nExternal Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(extDEDf_0[:, 8], bins = 150, 
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
    title = "Scaled\nExternal Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(ingestedDEDf_1[:, 8], bins = 150, 
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
    title = "Scaled\nIngested Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(ingestedDEDf_0[:, 8], bins = 150, 
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
    title = "Scaled\nIngested Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(fnaDEDf_1[:, 8], bins = 150, 
    subplot = 11, 
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
    title = "Scaled\nSmear MSI Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(fnaDEDf_0[:, 8], bins = 150, 
    subplot = 11, 
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
    title = "Scaled\nSmear MSI Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(diDEDf_1[:, 8], bins = 150, 
    subplot = 14, 
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
    title = "Scaled\nDirect Infusion Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(diDEDf_0[:, 8], bins = 150, 
    subplot = 14, 
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
    title = "Scaled\nDirect Infusion Dataset", 
    titlefont = font(12), 
    dpi = 300)


histogram!(trainDEFSDf_1[:, 8], bins = 150, 
    subplot = 3, 
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
    title = "Normalized\nTraining Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(trainDEFSDf_0[:, 8], bins = 150, 
    subplot = 3, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Relative MS Signal Intensity)", xguidefontsize=10, 
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
    title = "Normalized\nTraining Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(extDEFSDf_1[:, 8], bins = 150, 
    subplot = 6, 
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
    title = "Normalized\nExternal Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(extDEFSDf_0[:, 8], bins = 150, 
    subplot = 6, 
    framestyle = :box, 
    seriestype=:stephist, 
    xlabel = "Relative MS Signal Intensity)", xguidefontsize=10, 
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
    title = "Normalized\nExternal Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(ingestedDEFSDf_1[:, 8], bins = 150, 
    subplot = 9, 
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
    title = "Normalized\nIngested Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(ingestedDEFSDf_0[:, 8], bins = 150, 
    subplot = 9, 
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
    title = "Normalized\nIngested Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(fnaDEFSDf_1[:, 8], bins = 150, 
    subplot = 12, 
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
    title = "Normalized\nSmear MSI Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(fnaDEFSDf_0[:, 8], bins = 150, 
    subplot = 12, 
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
    title = "Normalized\nSmear MSI Dataset", 
    titlefont = font(12), 
    dpi = 300)
histogram!(diDEFSDf_1[:, 8], bins = 150, 
    subplot = 15, 
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
    title = "Normalized\nDirect Infusion Dataset", 
    titlefont = font(12), 
    dpi = 300)
    histogram!(diDEFSDf_0[:, 8], bins = 150, 
    subplot = 15, 
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
    title = "Normalized\nDirect Infusion Dataset", 
    titlefont = font(12), 
    dpi = 300)
  
## save ##
savefig(outplotTPTNdetaRiDistrution, "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\outplot738d5059_type0or1Distrution.png")
describe(trainDf[:, "738.5059"])
describe(trainDf[:, 2:18])
