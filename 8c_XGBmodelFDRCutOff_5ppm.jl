VERSION
## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
Pkg.build("PyCall")
Pkg.status()
using Random
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using PyCall
using StatsPlots
using Plots
using ScikitLearn  #: @sk_import, fit!, predict
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: GradientBoostingClassifier
@sk_import linear_model: LogisticRegression
@sk_import ensemble: RandomForestClassifier
@sk_import ensemble: AdaBoostClassifier
@sk_import tree: DecisionTreeClassifier
@sk_import metrics: recall_score
@sk_import neural_network: MLPClassifier
@sk_import svm: SVC
@sk_import neighbors: KNeighborsClassifier
@sk_import inspection: permutation_importance
#using ScikitLearn.GridSearch: RandomizedSearchCV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
#using ScikitLearn.GridSearch: GridSearchCV

## import packages from Python ##
jl = pyimport("joblib")             # used for loading models
f1_score = pyimport("sklearn.metrics").f1_score
matthews_corrcoef = pyimport("sklearn.metrics").matthews_corrcoef
make_scorer = pyimport("sklearn.metrics").make_scorer
f1 = make_scorer(f1_score, pos_label=1, average="binary")

## input ## predicted0n1, p(0), p(1)
trainDEFSDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_train24_std_0n1_pTP.csv", DataFrame)
extDEFSDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_ext24_std_0n1_pTP.csv", DataFrame)
fnaDEFSDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_FNA24_std_0n1_pTP.csv", DataFrame)

# ==================================================================================================
## prepare to plot confusion matrix for training set ##
trainDEFSDf[!, "CM"] .= String("")
    trainDEFSDf_TP = 0
    trainDEFSDf_FP = 0
    trainDEFSDf_TN = 0
    trainDEFSDf_FN = 0
    for i in 1:size(trainDEFSDf , 1)
        if (trainDEFSDf[i, "type"] == 1 && trainDEFSDf[i, "predicted0n1"] == 1)
            trainDEFSDf[i, "CM"] = "TP"
            trainDEFSDf_TP += 1
        elseif (trainDEFSDf[i, "type"] == 0 && trainDEFSDf[i, "predicted0n1"] == 1)
            trainDEFSDf[i, "CM"] = "FP"
            trainDEFSDf_FP += 1
        elseif (trainDEFSDf[i, "type"] == 0 && trainDEFSDf[i, "predicted0n1"] == 0)
            trainDEFSDf[i, "CM"] = "TN"
            trainDEFSDf_TN += 1
        elseif (trainDEFSDf[i, "type"] == 1 && trainDEFSDf[i, "predicted0n1"] == 0)
            trainDEFSDf[i, "CM"] = "FN"
            trainDEFSDf_FN += 1
        end
    end
    #
    CM_TrainWith = zeros(2, 2)
    CM_TrainWith[2, 1] = trainDEFSDf_TP
    CM_TrainWith[2, 2] = trainDEFSDf_FP
    CM_TrainWith[1, 2] = trainDEFSDf_TN
    CM_TrainWith[1, 1] = trainDEFSDf_FN

## save ##
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_train24_std_0n1_postPredict.csv"
CSV.write(savePath, trainDEFSDf)


# ==================================================================================================
## prepare to plot confusion matrix for Ext Val set ##
extDEFSDf[!, "CM"] .= String("")
    extDEFSDf_TP = 0
    extDEFSDf_FP = 0
    extDEFSDf_TN = 0
    extDEFSDf_FN = 0
    for i in 1:size(extDEFSDf , 1)
        if (extDEFSDf[i, "type"] == 1 && extDEFSDf[i, "predicted0n1"] == 1)
            extDEFSDf[i, "CM"] = "TP"
            extDEFSDf_TP += 1
        elseif (extDEFSDf[i, "type"] == 0 && extDEFSDf[i, "predicted0n1"] == 1)
            extDEFSDf[i, "CM"] = "FP"
            extDEFSDf_FP += 1
        elseif (extDEFSDf[i, "type"] == 0 && extDEFSDf[i, "predicted0n1"] == 0)
            extDEFSDf[i, "CM"] = "TN"
            extDEFSDf_TN += 1
        elseif (extDEFSDf[i, "type"] == 1 && extDEFSDf[i, "predicted0n1"] == 0)
            extDEFSDf[i, "CM"] = "FN"
            extDEFSDf_FN += 1
        end
    end
    #
    CM_ExtWith = zeros(2, 2)
    CM_ExtWith[2, 1] = extDEFSDf_TP
    CM_ExtWith[2, 2] = extDEFSDf_FP
    CM_ExtWith[1, 2] = extDEFSDf_TN
    CM_ExtWith[1, 1] = extDEFSDf_FN

## save ##
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_ext24_std_0n1_postPredict.csv"
CSV.write(savePath, extDEFSDf)

ingestedDEFSDf = vcat(trainDEFSDf, extDEFSDf)
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_ingested24_std_0n1_postPredict.csv"
CSV.write(savePath, ingestedDEFSDf)

# ==================================================================================================
## prepare to plot confusion matrix for FNA set ##
fnaDEFSDf[!, "CM"] .= String("")
    fnaDEFSDf_TP = 0
    fnaDEFSDf_FP = 0
    fnaDEFSDf_TN = 0
    fnaDEFSDf_FN = 0
    for i in 1:size(fnaDEFSDf , 1)
        if (fnaDEFSDf[i, "type"] == 1 && fnaDEFSDf[i, "predicted0n1"] == 1)
            fnaDEFSDf[i, "CM"] = "TP"
            fnaDEFSDf_TP += 1
        elseif (fnaDEFSDf[i, "type"] == 0 && fnaDEFSDf[i, "predicted0n1"] == 1)
            fnaDEFSDf[i, "CM"] = "FP"
            fnaDEFSDf_FP += 1
        elseif (fnaDEFSDf[i, "type"] == 0 && fnaDEFSDf[i, "predicted0n1"] == 0)
            fnaDEFSDf[i, "CM"] = "TN"
            fnaDEFSDf_TN += 1
        elseif (fnaDEFSDf[i, "type"] == 1 && fnaDEFSDf[i, "predicted0n1"] == 0)
            fnaDEFSDf[i, "CM"] = "FN"
            fnaDEFSDf_FN += 1
        end
    end
    #
    CM_FNAWith = zeros(2, 2)
    CM_FNAWith[2, 1] = fnaDEFSDf_TP
    CM_FNAWith[2, 2] = fnaDEFSDf_FP
    CM_FNAWith[1, 2] = fnaDEFSDf_TN
    CM_FNAWith[1, 1] = fnaDEFSDf_FN

## save ##
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_FNA24_std_0n1_postPredict.csv"
CSV.write(savePath, fnaDEFSDf)


# ==================================================================================================
## plot confusion matrix for training & testing sets ##
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()
TrainOutplotCM = plot(layout = layout, link = :both, 
        size = (2000, 600), margin = (10, :mm), dpi = 300)
heatmap!(["1", "0"], ["0", "1"], CM_TrainWith, cmap = :viridis, cbar = :true, 
        clims = (0, 45000), 
        subplot = 1, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "Training Dataset", 
        titlefont = font(16), 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n36,856"], subplot = 1)
        annotate!(["0"], ["1"], ["FP\n380"], subplot = 1, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n6,655"], subplot = 1, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n47,069"], subplot = 1)
heatmap!(["1", "0"], ["0", "1"], CM_ExtWith, cmap = :viridis, cbar = :true, 
        clims = (0, 3000), 
        subplot = 2, 
        framestyle = :box, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "External Validation Dataset", 
        titlefont = font(16), 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n2,586"], subplot = 2)
        annotate!(["0"], ["1"], ["FP\n55"], subplot = 2, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n546"], subplot = 2, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n2,888"], subplot = 2)
savefig(TrainOutplotCM, "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\prediction5ppm_trainExtValCM.png")


# ==================================================================================================
## plot confusion matrix for FNA set ##
default(grid = false, legend = false)
gr()
FNAOutplotCM = plot(size = (650, 600), margin = (10, :mm), dpi = 300)
heatmap!(["1", "0"], ["0", "1"], CM_FNAWith, cmap = :viridis, cbar = :true, 
        clims = (0, 40000), 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "Fine-Needle Aspiration Dataset", 
        titlefont = font(16), 
        size = (650,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n37,762"])
        annotate!(["0"], ["1"], ["FP\n26,534"])
        annotate!(["1"], ["0"], ["FN\n6,399"], font(color="white"))
        annotate!(["0"], ["0"], ["TN\n18,006"], font(color="white"))
savefig(FNAOutplotCM, "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\prediction5ppm_FNACM.png")


# ==================================================================================================
## prepare to plot P(TP)threshold-to-TPR curve ## training set
        sort!(trainDEFSDf, [:"p(1)"], rev = true)
        for i in 1:size(trainDEFSDf, 1)
            trainDEFSDf[i, "p(1)"] = round(float(trainDEFSDf[i, "p(1)"]), digits = 2)
        end
        #
        sort!(extDEFSDf, [:"p(1)"], rev = true)
        for i in 1:size(extDEFSDf, 1)
            extDEFSDf[i, "p(1)"] = round(float(extDEFSDf[i, "p(1)"]), digits = 2)
        end
        #
        sort!(fnaDEFSDf, [:"p(1)"], rev = true)
        for i in 1:size(fnaDEFSDf, 1)
            fnaDEFSDf[i, "p(1)"] = round(float(fnaDEFSDf[i, "p(1)"]), digits = 2)
        end
        #
        sort!(ingestedDEFSDf, [:"p(1)"], rev = true)
        for i in 1:size(ingestedDEFSDf, 1)
            ingestedDEFSDf[i, "p(1)"] = round(float(ingestedDEFSDf[i, "p(1)"]), digits = 2)
        end
    #
    ## define a function for Confusion Matrix ##
    function get1rate(df, thd)
        TP = 0  # 
        FN = 0  # 
        TN = 0  # 
        FP = 0  # 
        for i in 1:size(df , 1)
            if (df[i, "type"] == 1 && df[i, "p(1)"] >= thd)
                TP += 1
            elseif (df[i, "type"] == 1 && df[i, "p(1)"] < thd)
                FN += 1
            elseif (df[i, "type"] == 0 && df[i, "p(1)"] >= thd)
                FP += 1
            elseif (df[i, "type"] == 0 && df[i, "p(1)"] < thd)
                TN += 1
            end
        end
        return (TP / (TP + FN)), (FN / (TP + FN)), (FP / (FP + TP)), (FP / (FP + TN)), (TN / (TN + FP))
    end
    #
    ## call function and insert arrays as columns ##
    Train_TPR = []
    Train_FNR = []
    Train_FDR = []
    Train_FPR = []
    Train_TNR = []
    prob = -1
    TPR = 0
    FNR = 0
    FDR = 0
    FPR = 0
    TNR = 0
    for temp in Array(trainDEFSDf[:, "p(1)"])
        if (temp != prob)
            println(temp)
            prob = temp
            TPR, FNR, FDR, FPR, TNR = get1rate(trainDEFSDf, prob)
            push!(Train_TPR, TPR)
            push!(Train_FNR, FNR)
            push!(Train_FDR, FDR)
            push!(Train_FPR, FPR)
            push!(Train_TNR, TNR)
        else
            push!(Train_TPR, TPR)
            push!(Train_FNR, FNR)
            push!(Train_FDR, FDR)
            push!(Train_FPR, FPR)
            push!(Train_TNR, TNR)
        end
    end
    trainDEFSDf[!, "TPR"] = Train_TPR
    trainDEFSDf[!, "FNR"] = Train_FNR
    trainDEFSDf[!, "FDR"] = Train_FDR
    trainDEFSDf[!, "FPR"] = Train_FPR
    trainDEFSDf[!, "TNR"] = Train_TNR

## save ##
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\postPredict5ppm_tPRfNRfDR_train.csv"
CSV.write(savePath, trainDEFSDf)


# ==================================================================================================
## prepare to plot P(TP)threshold-to-TPR curve ## Ext Val set
    ## call function and insert arrays as columns ##
    Ext_TPR = []
    Ext_FNR = []
    Ext_FDR = []
    Ext_FPR = []
    Ext_TNR = []
    prob = -1
    TPR = 0
    FNR = 0
    FDR = 0
    FPR = 0
    TNR = 0
    for temp in Array(extDEFSDf[:, "p(1)"])
        if (temp != prob)
            println(temp)
            prob = temp
            TPR, FNR, FDR, FPR, TNR = get1rate(extDEFSDf, prob)
            push!(Ext_TPR, TPR)
            push!(Ext_FNR, FNR)
            push!(Ext_FDR, FDR)
            push!(Ext_FPR, FPR)
            push!(Ext_TNR, TNR)
        else
            push!(Ext_TPR, TPR)
            push!(Ext_FNR, FNR)
            push!(Ext_FDR, FDR)
            push!(Ext_FPR, FPR)
            push!(Ext_TNR, TNR)
        end
    end
    extDEFSDf[!, "TPR"] = Ext_TPR
    extDEFSDf[!, "FNR"] = Ext_FNR
    extDEFSDf[!, "FDR"] = Ext_FDR
    extDEFSDf[!, "FPR"] = Ext_FPR
    extDEFSDf[!, "TNR"] = Ext_TNR

## save ##
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\postPredict5ppm_tPRfNRfDR_ext.csv"
CSV.write(savePath, extDEFSDf)


# ==================================================================================================
## prepare to plot P(TP)threshold-to-TPR curve ## FNA set
    ## call function and insert arrays as columns ##
    FNA_TPR = []
    FNA_FNR = []
    FNA_FDR = []
    FNA_FPR = []
    FNA_TNR = []
    prob = -1
    TPR = 0
    FNR = 0
    FDR = 0
    FPR = 0
    TNR = 0
    for temp in Array(fnaDEFSDf[:, "p(1)"])
        if (temp != prob)
            println(temp)
            prob = temp
            TPR, FNR, FDR, FPR, TNR = get1rate(fnaDEFSDf, prob)
            push!(FNA_TPR, TPR)
            push!(FNA_FNR, FNR)
            push!(FNA_FDR, FDR)
            push!(FNA_FPR, FPR)
            push!(FNA_TNR, TNR)
        else
            push!(FNA_TPR, TPR)
            push!(FNA_FNR, FNR)
            push!(FNA_FDR, FDR)
            push!(FNA_FPR, FPR)
            push!(FNA_TNR, TNR)
        end
    end
    fnaDEFSDf[!, "TPR"] = FNA_TPR
    fnaDEFSDf[!, "FNR"] = FNA_FNR
    fnaDEFSDf[!, "FDR"] = FNA_FDR
    fnaDEFSDf[!, "FPR"] = FNA_FPR
    fnaDEFSDf[!, "TNR"] = FNA_TNR

## save ##
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\postPredict5ppm_tPRfNRfDR_FNA.csv"
CSV.write(savePath, fnaDEFSDf)


# ==================================================================================================
## plot P(1)threshold-to-TPR & P(1)threshold-to-TNR ## for training set
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()
TrainOutplotP1toRate = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)
plot!(trainDEFSDf[:, end-6], [trainDEFSDf[:, end-4] trainDEFSDf[:, end-3]], 
        subplot = 1, framestyle = :box, 
        xlabel = "P(1) Threshold", 
        xguidefontsize=12, 
        label = ["True Positive Rate" "False Negative Rate"], 
        xtickfontsize = 10, 
        ytickfontsize= 10, 
        legend = :left, 
        legendfont = font(10), 
        size = (1200,600), 
        dpi = 300)
plot!(trainDEFSDf[:, end-6], trainDEFSDf[:, end-2], 
        subplot = 2, framestyle = :box, 
        xlabel = "P(1) Threshold", 
        xguidefontsize=12, 
        label = "False Discovery Rate", 
        yguidefontsize=12, 
        xtickfontsize = 10, 
        ytickfontsize= 10, 
        ylims = [0, 0.5], 
        legend = :best, 
        legendfont = font(10), 
        size = (1200,600), 
        dpi = 300)
        new_yticks = ([0.05], ["\$\\bar"], ["purple"])
        new_yticks2 = ([0.10], ["\$\\bar"], ["red"])
        hline!(new_yticks[1], label = "5% FDR-Controlled Cutoff at P(1) = 1.00", legendfont = font(10), lc = "purple", subplot = 2)
        hline!(new_yticks2[1], label = "10% FDR-Controlled Cutoff at P(1) = 1.00", legendfont = font(10), lc = "red", subplot = 2)
savefig(TrainOutplotP1toRate, "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\prediction5ppm_P1threshold2TPRFNRFDR.png")

