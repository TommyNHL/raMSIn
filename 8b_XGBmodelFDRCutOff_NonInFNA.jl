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
fnaDEFSDf = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_NonIn\\df_NonInFNA24_allstd_0n1_pTP.csv", DataFrame)

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
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_NonIn\\df_NonInFNA24_allstd_0n1_postPredict.csv"
CSV.write(savePath, fnaDEFSDf)


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
        annotate!(["1"], ["1"], ["TP\n9,758"], font(color="white"))
        annotate!(["0"], ["1"], ["FP\n9,296"], font(color="white"))
        annotate!(["1"], ["0"], ["FN\n544"], font(color="white"))
        annotate!(["0"], ["0"], ["TN\n35,244"])
savefig(FNAOutplotCM, "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_NonIn\\prediction_NonInFNACM_all.png")


# ==================================================================================================
## prepare to plot P(TP)threshold-to-TPR curve ## training set
        #
        sort!(fnaDEFSDf, [:"p(1)"], rev = true)
        for i in 1:size(fnaDEFSDf, 1)
            fnaDEFSDf[i, "p(1)"] = round(float(fnaDEFSDf[i, "p(1)"]), digits = 2)
        end
        #
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
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_NonIn\\postPredict_tPRfNRfDR_NonInFNA_all.csv"
CSV.write(savePath, fnaDEFSDf)


# ==================================================================================================
## plot P(1)threshold-to-TPR & P(1)threshold-to-TNR ## for training set
layout = @layout [a{0.50w,1.0h} b{0.50w,1.0h}]
default(grid = false, legend = false)
gr()
FNAOutplotP1toRate = plot(layout = layout, link = :both, 
        size = (1200, 600), margin = (8, :mm), dpi = 300)
plot!(fnaDEFSDf[:, end-6], [fnaDEFSDf[:, end-4] fnaDEFSDf[:, end-3]], 
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
plot!(fnaDEFSDf[:, end-6], fnaDEFSDf[:, end-2], 
        subplot = 2, framestyle = :box, 
        xlabel = "P(1) Threshold", 
        xguidefontsize=12, 
        label = "False Discovery Rate", 
        yguidefontsize=12, 
        xtickfontsize = 10, 
        ytickfontsize= 10, 
        ylims = [0, 1], 
        legend = :best, 
        legendfont = font(10), 
        size = (1200,600), 
        dpi = 300)
        new_yticks = ([0.05], ["\$\\bar"], ["purple"])
        new_yticks2 = ([0.10], ["\$\\bar"], ["red"])
        hline!(new_yticks[1], label = "5% FDR-Controlled Cutoff at P(1) = 1.00", legendfont = font(10), lc = "purple", subplot = 2)
        hline!(new_yticks2[1], label = "10% FDR-Controlled Cutoff at P(1) = 1.00", legendfont = font(10), lc = "red", subplot = 2)
savefig(FNAOutplotP1toRate, "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_NonIn\\prediction_P1threshold2TPRFNRFDR_all.png")

