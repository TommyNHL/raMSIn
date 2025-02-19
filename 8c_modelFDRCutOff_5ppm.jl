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
@sk_import ensemble: VotingClassifier
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

## input ## predicted0n1
trainDEFSDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\afterModelSelection\\dbMSIn_5ppm\\df_train_dbMSIn5ppm_norm_0n1.csv", DataFrame)
extDEFSDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\afterModelSelection\\dbMSIn_5ppm\\df_ext_dbMSIn5ppm_norm_0n1.csv", DataFrame)
fnaDEFSDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\afterModelSelection\\dbMSIn_5ppm\\df_FNA_dbMSIn5ppm_norm_0n1.csv", DataFrame)
diDEFSDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\afterModelSelection\\dbMSIn_5ppm\\df_Di_dbMSIn5ppm_norm_0n1.csv", DataFrame)

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
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\afterModelSelection\\dbMSIn_5ppm\\df_train_dbMSIn5ppm_norm_0n1_postPredict.csv"
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
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\afterModelSelection\\dbMSIn_5ppm\\df_ext_dbMSIn5ppm_norm_0n1_postPredict.csv"
CSV.write(savePath, extDEFSDf)

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
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\afterModelSelection\\dbMSIn_5ppm\\df_FNA_dbMSIn5ppm_norm_0n1_postPredict.csv"
CSV.write(savePath, fnaDEFSDf)


# ==================================================================================================
## prepare to plot confusion matrix for DirectInfusion set ##
diDEFSDf[!, "CM"] .= String("")
    diDEFSDf_TP = 0
    diDEFSDf_FP = 0
    diDEFSDf_TN = 0
    diDEFSDf_FN = 0
    for i in 1:size(diDEFSDf , 1)
        if (diDEFSDf[i, "type"] == 1 && diDEFSDf[i, "predicted0n1"] == 1)
            diDEFSDf[i, "CM"] = "TP"
            diDEFSDf_TP += 1
        elseif (diDEFSDf[i, "type"] == 0 && diDEFSDf[i, "predicted0n1"] == 1)
            diDEFSDf[i, "CM"] = "FP"
            diDEFSDf_FP += 1
        elseif (diDEFSDf[i, "type"] == 0 && diDEFSDf[i, "predicted0n1"] == 0)
            diDEFSDf[i, "CM"] = "TN"
            diDEFSDf_TN += 1
        elseif (diDEFSDf[i, "type"] == 1 && diDEFSDf[i, "predicted0n1"] == 0)
            diDEFSDf[i, "CM"] = "FN"
            diDEFSDf_FN += 1
        end
    end
    #
    CM_DiWith = zeros(2, 2)
    CM_DiWith[2, 1] = diDEFSDf_TP
    CM_DiWith[2, 2] = diDEFSDf_FP
    CM_DiWith[1, 2] = diDEFSDf_TN
    CM_DiWith[1, 1] = diDEFSDf_FN

## save ##
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\afterModelSelection\\dbMSIn_5ppm\\df_Di_dbMSIn5ppm_norm_0n1_postPredict.csv"
CSV.write(savePath, diDEFSDf)


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
        annotate!(["1"], ["1"], ["TP\n39,424"], subplot = 1)
        annotate!(["0"], ["1"], ["FP\n6,900"], subplot = 1, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n4,087"], subplot = 1, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n40,549"], subplot = 1)
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
        annotate!(["1"], ["1"], ["TP\n2,787"], subplot = 2)
        annotate!(["0"], ["1"], ["FP\n495"], subplot = 2, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n345"], subplot = 2, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n2,448"], subplot = 2)
savefig(TrainOutplotCM, "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\afterModelSelection\\dbMSIn_5ppm\\prediction_trainExtValCM.png")


# ==================================================================================================
## plot confusion matrix for FNA & DirectInfusion set ##
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()
FNAOutplotCM = plot(layout = layout, link = :both, 
        size = (2000, 600), margin = (10, :mm), dpi = 300) 
heatmap!(["1", "0"], ["0", "1"], CM_FNAWith, cmap = :viridis, cbar = :true, 
        clims = (0, 40000), 
        subplot = 1, 
        xlabel = "Expected", xguidefontsize=16, 
        ylabel = "Predicted", yguidefontsize=16, 
        xtickfontsize = 12, 
        ytickfontsize= 12, 
        title = "Fine-Needle Aspiration Dataset", 
        titlefont = font(16), 
        size = (1400,600), 
        dpi = 300)
        annotate!(["1"], ["1"], ["TP\n32,981"], subplot = 1)
        annotate!(["0"], ["1"], ["FP\n14,547"], subplot = 1, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n11,180"], subplot = 1, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n29,993"], subplot = 1)
heatmap!(["1", "0"], ["0", "1"], CM_DiWith, cmap = :viridis, cbar = :true, 
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
        annotate!(["1"], ["1"], ["TP\n1,803"], subplot = 2)
        annotate!(["0"], ["1"], ["FP\n1,079"], subplot = 2, font(color="white"))
        annotate!(["1"], ["0"], ["FN\n1,227"], subplot = 2, font(color="white"))
        annotate!(["0"], ["0"], ["TN\n1,948"], subplot = 2)
savefig(FNAOutplotCM, "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\afterModelSelection\\dbMSIn_5ppm\\prediction_FNAnDICM.png")
