VERSION
## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\T1208\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
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
@sk_import svm: LinearSVC
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

## input training set ## 90960 x 20 df
trainDEFSDf = CSV.read("G:\\raMSIn\\XGB_Importance2\\df_train24_std.csv", DataFrame)
trainDEFSDf[trainDEFSDf.type .== 1, :]
    ## calculate weight ## 0: 47449, 1: 43511
    Yy_train = deepcopy(trainDEFSDf[:, end])  # 0.9585; 1.0453
    sampleW = []
    for w in Vector(Yy_train)
        if w == 0
            push!(sampleW, 0.9585)
        elseif w == 1
            push!(sampleW, 1.0453)
        end
    end 

## input ext val set ## 6075 x 20 df
extDEFSDf = CSV.read("G:\\raMSIn\\XGB_Importance2\\df_ext24_std.csv", DataFrame)
extDEFSDf[extDEFSDf.type .== 1, :]
    ## calculate weight ## 0: 2943, 1: 3132
    Yy_ext = deepcopy(extDEFSDf[:, end])  # 1.0321; 0.9698
    sampleExtW = []
    for w in Vector(Yy_ext)
        if w == 0
            push!(sampleExtW, 1.0321)
        elseif w == 1
            push!(sampleExtW, 0.9698)
        end
    end 

## reconstruct a whole set ## 97035 x 20 df
ingestedDEFSDf = vcat(trainDEFSDf, extDEFSDf)
ingestedDEFSDf[ingestedDEFSDf.type .== 1, :]
    ## calculate weight ## 0: 50392, 1: 46643
    Yy_ingested = deepcopy(ingestedDEFSDf[:, end])  # 0.9628; 1.0402
    sampleIngestedW = []
    for w in Vector(Yy_ingested)
        if w == 0
            push!(sampleIngestedW, 0.9628)
        elseif w == 1
            push!(sampleIngestedW, 1.0402)
        end
    end 

## input FNA set ## 88701 x 20 df
fnaDEFSDf = CSV.read("G:\\raMSIn\\XGB_Importance2\\df_FNA24_std.csv", DataFrame)
fnaDEFSDf[fnaDEFSDf.type .== 1, :]
    ## calculate weight ##  0: 44540, 1: 44161
    Yy_FNA = deepcopy(fnaDEFSDf[:, end])  # 0.9957; 1.0043
    sampleFNAW = []
    for w in Vector(Yy_FNA)
        if w == 0
            push!(sampleFNAW, 0.9957)
        elseif w == 1
            push!(sampleFNAW, 1.0043)
        end
    end 


## define functions for performace evaluation ##
    # Maximum absolute error
    # mean square error (MSE) calculation
    # Root mean square error (RMSE) calculation
    function errorDetermination(arrRi, predictedRi)
        sumAE = 0
        maxAE = 0
        for i = 1:size(predictedRi, 1)
            AE = abs(arrRi[i] - predictedRi[i])
            if (AE > maxAE)
                maxAE = AE
            end
            sumAE += (AE ^ 2)
        end
        MSE = sumAE / size(predictedRi, 1)
        RMSE = MSE ^ 0.5
        return maxAE, MSE, RMSE
    end
    #
    # Average score
    function avgScore(arrAcc, cv)
        sumAcc = 0
        for acc in arrAcc
            sumAcc += acc
        end
        return sumAcc / cv
    end


# ==================================================================================================
## define a function for Random Forest ##
function optimRandomForestClass(inputDB, inputDB_ingested, inputDB_ext, inputDB_FNA)
    #leaf_r = vcat(2, 4, 8, 12, 18)  # 5
    leaf_r = vcat(collect(20:2:30))  # 6
    #depth_r = vcat(collect(2:1:10))  # 9
    depth_r = vcat(2, 4, 8, collect(12:1:14))  # 6
    #split_r = vcat(collect(2:1:10))  # 9
    split_r = vcat(collect(10:10:20))  # 2
    tree_r = vcat(collect(50:50:300))  # 6

    rs = 42
    z = zeros(1,49)
    itr = 1

    N_train = inputDB
    M_train = inputDB_ingested
    M_ext = inputDB_ext
    M_FNA = inputDB_FNA

    for l in leaf_r
        for d in depth_r
            for s in split_r
                for t in tree_r
                    println("itr=", itr, ", leaf=", l, ", depth=", d, ", minSsplit=", s, ", tree=", t)
                    println("## loading in data ##")
                    Xx_train = deepcopy(M_train[:, 2:end-1])
                    nn_train = deepcopy(N_train[:, 2:end-1])
                    Xx_Ext = deepcopy(M_ext[:, 2:end-1])
                    Xx_FNA = deepcopy(M_FNA[:, 2:end-1])
                    #
                    Yy_train = deepcopy(M_train[:, end])
                    mm_train = deepcopy(N_train[:, end])
                    Yy_Ext = deepcopy(M_ext[:, end])
                    Yy_FNA = deepcopy(M_FNA[:, end])
                    println("## Classification ##")
                    reg = RandomForestClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=s, n_jobs=-1, oob_score =true, random_state=rs, class_weight=Dict(0=>0.9628, 1=>1.0402))
                    println("## fit ##")
                    fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                    importances = permutation_importance(reg, Matrix(Xx_FNA), Vector(Yy_FNA), n_repeats=10, random_state=42, n_jobs=-1)
                    print(importances["importances_mean"])
                    if itr == 1
                        z[1,1] = l
                        z[1,2] = t
                        z[1,3] = d
                        z[1,4] = s
                        z[1,5] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                        z[1,6] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                        z[1,7] = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                        z[1,8] = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                        println("## CV ##")
                        f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                        z[1,9] = avgScore(f1_10_train, 3)
                        z[1,10] = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        z[1,11] = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        z[1,12] = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                        z[1,13] = rs
                        z[1,14] = importances["importances_mean"][1]
                        z[1,15] = importances["importances_mean"][2]
                        z[1,16] = importances["importances_mean"][3]
                        z[1,17] = importances["importances_mean"][4]
                        z[1,18] = importances["importances_mean"][5]
                        z[1,19] = importances["importances_mean"][6]
                        z[1,20] = importances["importances_mean"][7]
                        z[1,21] = importances["importances_mean"][8]
                        z[1,22] = importances["importances_mean"][9]
                        z[1,23] = importances["importances_mean"][10]
                        z[1,24] = importances["importances_mean"][11]
                        z[1,25] = importances["importances_mean"][12]
                        z[1,26] = importances["importances_mean"][13]
                        z[1,27] = importances["importances_mean"][14]
                        z[1,28] = importances["importances_mean"][15]
                        z[1,29] = importances["importances_mean"][16]
                        z[1,30] = importances["importances_mean"][17]
                        z[1,31] = importances["importances_mean"][18]
                        z[1,32] = importances["importances_std"][1]
                        z[1,33] = importances["importances_std"][2]
                        z[1,34] = importances["importances_std"][3]
                        z[1,35] = importances["importances_std"][4]
                        z[1,36] = importances["importances_std"][5]
                        z[1,37] = importances["importances_std"][6]
                        z[1,38] = importances["importances_std"][7]
                        z[1,39] = importances["importances_std"][8]
                        z[1,40] = importances["importances_std"][9]
                        z[1,41] = importances["importances_std"][10]
                        z[1,42] = importances["importances_std"][11]
                        z[1,43] = importances["importances_std"][12]
                        z[1,44] = importances["importances_std"][13]
                        z[1,45] = importances["importances_std"][14]
                        z[1,46] = importances["importances_std"][15]
                        z[1,47] = importances["importances_std"][16]
                        z[1,48] = importances["importances_std"][17]
                        z[1,49] = importances["importances_std"][18]
                    else
                        itrain = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                        jtrain = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                        ival = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                        jval = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                        println("## CV ##")
                        f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                        traincvtrain = avgScore(f1_10_train, 3) 
                        f1s = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        mccs = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        rec = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                        im1 = importances["importances_mean"][1]
                        im2 = importances["importances_mean"][2]
                        im3 = importances["importances_mean"][3]
                        im4 = importances["importances_mean"][4]
                        im5 = importances["importances_mean"][5]
                        im6 = importances["importances_mean"][6]
                        im7 = importances["importances_mean"][7]
                        im8 = importances["importances_mean"][8]
                        im9 = importances["importances_mean"][9]
                        im10 = importances["importances_mean"][10]
                        im11 = importances["importances_mean"][11]
                        im12 = importances["importances_mean"][12]
                        im13 = importances["importances_mean"][13]
                        im14 = importances["importances_mean"][14]
                        im15 = importances["importances_mean"][15]
                        im16 = importances["importances_mean"][16]
                        im17 = importances["importances_mean"][17]
                        im18 = importances["importances_mean"][18]
                        sd1 = importances["importances_std"][1]
                        sd2 = importances["importances_std"][2]
                        sd3 = importances["importances_std"][3]
                        sd4 = importances["importances_std"][4]
                        sd5 = importances["importances_std"][5]
                        sd6 = importances["importances_std"][6]
                        sd7 = importances["importances_std"][7]
                        sd8 = importances["importances_std"][8]
                        sd9 = importances["importances_std"][9]
                        sd10 = importances["importances_std"][10]
                        sd11 = importances["importances_std"][11]
                        sd12 = importances["importances_std"][12]
                        sd13 = importances["importances_std"][13]
                        sd14 = importances["importances_std"][14]
                        sd15 = importances["importances_std"][15]
                        sd16 = importances["importances_std"][16]
                        sd17 = importances["importances_std"][17]
                        sd18 = importances["importances_std"][18]
                        z = vcat(z, [l t d s itrain jtrain ival jval traincvtrain f1s mccs rec rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 im11 im12 im13 im14 im15 im16 im17 im18 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10 sd11 sd12 sd13 sd14 sd15 sd16 sd17 sd18])
                        println(z[end, :])
                    end
                    println("End of ", itr, " iterations")
                    itr += 1
                end
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], depth = z[:,3], minSplit = z[:,4], f1_train = z[:,5], mcc_train = z[:,6], f1_ext = z[:,7], mcc_ext = z[:,8], f1_3Ftrain = z[:,9], f1_fna = z[:,10], mcc_fna = z[:,11], recall = z[:,12], state = z[:,13], im1 = z[:,14], im2 = z[:,15], im3 = z[:,16], im4 = z[:,17], im5 = z[:,18], im6 = z[:,19], im7 = z[:,20], im8 = z[:,21], im9 = z[:,22], im10 = z[:,23], im11 = z[:,24], im12 = z[:,25], im13 = z[:,26], im14 = z[:,27], im15 = z[:,28], im16 = z[:,29], im17 = z[:,30], im18 = z[:,31], sd1 = z[:,32], sd2 = z[:,33], sd3 = z[:,34], sd4 = z[:,35], sd5 = z[:,36], sd6 = z[:,37], sd7 = z[:,38], sd8 = z[:,39], sd9 = z[:,40], sd10 = z[:,41], sd11 = z[:,42], sd12 = z[:,43], sd13 = z[:,44], sd14 = z[:,45], sd15 = z[:,46], sd16 = z[:,47], sd17 = z[:,48], sd18 = z[:,49])
    z_df_sorted = sort(z_df, [:recall, :f1_fna, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

## call Random Forest ##
optiSearch_df = optimRandomForestClass(trainDEFSDf, ingestedDEFSDf, extDEFSDf, fnaDEFSDf)

## save ##
savePath = "H:\\3_output_raMSIn\\hyperparameterTuning_modelSelection_RF2.csv"
CSV.write(savePath, optiSearch_df)


# ==================================================================================================
## define a function for Decision Tree ##
function optimDecisionTreeClass(inputDB, inputDB_ingested, inputDB_ext, inputDB_FNA)
    #leaf_r = vcat(2, 4, 8, 12, 18)  # 5
    leaf_r = vcat(collect(11:2:19), collect(20:2:40), collect(45:5:80), 100, 200, 500)  # 5+11+8+3=27
    #depth_r = vcat(collect(2:1:10))  # 9
    depth_r = vcat(collect(2:1:14))  # 13
    split_r = vcat(collect(2:1:10))  # 9

    rs = 42
    z = zeros(1,48)
    itr = 1

    N_train = inputDB
    M_train = inputDB_ingested
    M_ext = inputDB_ext
    M_FNA = inputDB_FNA

    for l in leaf_r
        for d in depth_r
            for s in split_r
                println("itr=", itr, ", leaf=", l, ", depth=", d, ", minSsplit=", s)
                println("## loading in data ##")
                Xx_train = deepcopy(M_train[:, 2:end-1])
                nn_train = deepcopy(N_train[:, 2:end-1])
                Xx_Ext = deepcopy(M_ext[:, 2:end-1])
                Xx_FNA = deepcopy(M_FNA[:, 2:end-1])
                #
                Yy_train = deepcopy(M_train[:, end])
                mm_train = deepcopy(N_train[:, end])
                Yy_Ext = deepcopy(M_ext[:, end])
                Yy_FNA = deepcopy(M_FNA[:, end])
                println("## Classification ##")
                reg = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, min_samples_split=s, random_state=rs, class_weight=Dict(0=>0.9628, 1=>1.0402))
                println("## fit ##")
                fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                importances = permutation_importance(reg, Matrix(Xx_FNA), Vector(Yy_FNA), n_repeats=10, random_state=42)
                print(importances["importances_mean"])
                if itr == 1
                    z[1,1] = l
                    z[1,2] = d
                    z[1,3] = s
                    z[1,4] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,5] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,6] = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                    z[1,7] = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    z[1,8] = avgScore(f1_10_train, 3)
                    z[1,9] = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    z[1,10] = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    z[1,11] = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                    z[1,12] = rs
                    z[1,13] = importances["importances_mean"][1]
                    z[1,14] = importances["importances_mean"][2]
                    z[1,15] = importances["importances_mean"][3]
                    z[1,16] = importances["importances_mean"][4]
                    z[1,17] = importances["importances_mean"][5]
                    z[1,18] = importances["importances_mean"][6]
                    z[1,19] = importances["importances_mean"][7]
                    z[1,20] = importances["importances_mean"][8]
                    z[1,21] = importances["importances_mean"][9]
                    z[1,22] = importances["importances_mean"][10]
                    z[1,23] = importances["importances_mean"][11]
                    z[1,24] = importances["importances_mean"][12]
                    z[1,25] = importances["importances_mean"][13]
                    z[1,26] = importances["importances_mean"][14]
                    z[1,27] = importances["importances_mean"][15]
                    z[1,28] = importances["importances_mean"][16]
                    z[1,29] = importances["importances_mean"][17]
                    z[1,30] = importances["importances_mean"][18]
                    z[1,31] = importances["importances_std"][1]
                    z[1,32] = importances["importances_std"][2]
                    z[1,33] = importances["importances_std"][3]
                    z[1,34] = importances["importances_std"][4]
                    z[1,35] = importances["importances_std"][5]
                    z[1,36] = importances["importances_std"][6]
                    z[1,37] = importances["importances_std"][7]
                    z[1,38] = importances["importances_std"][8]
                    z[1,39] = importances["importances_std"][9]
                    z[1,40] = importances["importances_std"][10]
                    z[1,41] = importances["importances_std"][11]
                    z[1,42] = importances["importances_std"][12]
                    z[1,43] = importances["importances_std"][13]
                    z[1,44] = importances["importances_std"][14]
                    z[1,45] = importances["importances_std"][15]
                    z[1,46] = importances["importances_std"][16]
                    z[1,47] = importances["importances_std"][17]
                    z[1,48] = importances["importances_std"][18]
                else
                    itrain = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    jtrain = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    ival = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                    jval = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    traincvtrain = avgScore(f1_10_train, 3) 
                    f1s = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    mccs = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    rec = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                    im1 = importances["importances_mean"][1]
                    im2 = importances["importances_mean"][2]
                    im3 = importances["importances_mean"][3]
                    im4 = importances["importances_mean"][4]
                    im5 = importances["importances_mean"][5]
                    im6 = importances["importances_mean"][6]
                    im7 = importances["importances_mean"][7]
                    im8 = importances["importances_mean"][8]
                    im9 = importances["importances_mean"][9]
                    im10 = importances["importances_mean"][10]
                    im11 = importances["importances_mean"][11]
                    im12 = importances["importances_mean"][12]
                    im13 = importances["importances_mean"][13]
                    im14 = importances["importances_mean"][14]
                    im15 = importances["importances_mean"][15]
                    im16 = importances["importances_mean"][16]
                    im17 = importances["importances_mean"][17]
                    im18 = importances["importances_mean"][18]
                    sd1 = importances["importances_std"][1]
                    sd2 = importances["importances_std"][2]
                    sd3 = importances["importances_std"][3]
                    sd4 = importances["importances_std"][4]
                    sd5 = importances["importances_std"][5]
                    sd6 = importances["importances_std"][6]
                    sd7 = importances["importances_std"][7]
                    sd8 = importances["importances_std"][8]
                    sd9 = importances["importances_std"][9]
                    sd10 = importances["importances_std"][10]
                    sd11 = importances["importances_std"][11]
                    sd12 = importances["importances_std"][12]
                    sd13 = importances["importances_std"][13]
                    sd14 = importances["importances_std"][14]
                    sd15 = importances["importances_std"][15]
                    sd16 = importances["importances_std"][16]
                    sd17 = importances["importances_std"][17]
                    sd18 = importances["importances_std"][18]
                    z = vcat(z, [l d s itrain jtrain ival jval traincvtrain f1s mccs rec rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 im11 im12 im13 im14 im15 im16 im17 im18 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10 sd11 sd12 sd13 sd14 sd15 sd16 sd17 sd18])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], depth = z[:,2], minSplit = z[:,3], f1_train = z[:,4], mcc_train = z[:,5], f1_ext = z[:,6], mcc_ext = z[:,7], f1_3Ftrain = z[:,8], f1_fna = z[:,9], mcc_fna = z[:,10], recall = z[:,11], state = z[:,12], im1 = z[:,13], im2 = z[:,14], im3 = z[:,15], im4 = z[:,16], im5 = z[:,17], im6 = z[:,18], im7 = z[:,19], im8 = z[:,20], im9 = z[:,21], im10 = z[:,22], im11 = z[:,23], im12 = z[:,24], im13 = z[:,25], im14 = z[:,26], im15 = z[:,27], im16 = z[:,28], im17 = z[:,29], im18 = z[:,30], sd1 = z[:,31], sd2 = z[:,32], sd3 = z[:,33], sd4 = z[:,34], sd5 = z[:,35], sd6 = z[:,36], sd7 = z[:,37], sd8 = z[:,38], sd9 = z[:,39], sd10 = z[:,40], sd11 = z[:,41], sd12 = z[:,42], sd13 = z[:,43], sd14 = z[:,44], sd15 = z[:,45], sd16 = z[:,46], sd17 = z[:,47], sd18 = z[:,48])
    z_df_sorted = sort(z_df, [:recall, :f1_fna, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

## call Decision Tree ##
optiSearch_df = optimDecisionTreeClass(trainDEFSDf, ingestedDEFSDf, extDEFSDf, fnaDEFSDf)

## save ##
savePath = "H:\\3_output_raMSIn\\hyperparameterTuning_modelSelection_DT2.csv"
CSV.write(savePath, optiSearch_df)


# ==================================================================================================
## define a function for Logistic Regression ##
function optimLR(inputDB, inputDB_ingested, inputDB_ext, inputDB_FNA)
    penalty_r = ["l1", "l2"]  # 2
    solver_rs = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]  # 5
    #c_values_r = vcat(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01)  # 14
    #c_values_r = vcat(0.02, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075, 0.0005)  # 8
    c_values_r = vcat(0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.000075, 0.00005, 0.000025, 0.00001)  # 13

    rs = 42
    z = zeros(1,48)
    itr = 1
    pnn = 0

    N_train = inputDB
    M_train = inputDB_ingested
    M_ext = inputDB_ext
    M_FNA = inputDB_FNA

    for pn in 1:2
        for s in 1:5
            for c in c_values_r
                if solver_rs[s] == "lbfgs" || solver_rs[s] == "newton-cg" || solver_rs[s] == "sag"
                    pnn = 2
                else
                    pnn = pn
                end
                println("## loading in data ##")
                Xx_train = deepcopy(M_train[:, 2:end-1])
                nn_train = deepcopy(N_train[:, 2:end-1])
                Xx_Ext = deepcopy(M_ext[:, 2:end-1])
                Xx_FNA = deepcopy(M_FNA[:, 2:end-1])
                #
                Yy_train = deepcopy(M_train[:, end])
                mm_train = deepcopy(N_train[:, end])
                Yy_Ext = deepcopy(M_ext[:, end])
                Yy_FNA = deepcopy(M_FNA[:, end])
                println("## Classification ##")
                reg = LogisticRegression(penalty=penalty_r[pnn], C=c, solver=solver_rs[s], max_iter=5000, random_state=rs, class_weight=Dict(0=>0.9628, 1=>1.0402))
                println("## fit ##")
                fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                importances = permutation_importance(reg, Matrix(Xx_FNA), Vector(Yy_FNA), n_repeats=10, random_state=42)
                print(importances["importances_mean"])
                if itr == 1
                    z[1,1] = pn
                    z[1,2] = s
                    z[1,3] = c
                    z[1,4] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,5] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,6] = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                    z[1,7] = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    z[1,8] = avgScore(f1_10_train, 3)
                    z[1,9] = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    z[1,10] = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    z[1,11] = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                    z[1,12] = rs
                    z[1,13] = importances["importances_mean"][1]
                    z[1,14] = importances["importances_mean"][2]
                    z[1,15] = importances["importances_mean"][3]
                    z[1,16] = importances["importances_mean"][4]
                    z[1,17] = importances["importances_mean"][5]
                    z[1,18] = importances["importances_mean"][6]
                    z[1,19] = importances["importances_mean"][7]
                    z[1,20] = importances["importances_mean"][8]
                    z[1,21] = importances["importances_mean"][9]
                    z[1,22] = importances["importances_mean"][10]
                    z[1,23] = importances["importances_mean"][11]
                    z[1,24] = importances["importances_mean"][12]
                    z[1,25] = importances["importances_mean"][13]
                    z[1,26] = importances["importances_mean"][14]
                    z[1,27] = importances["importances_mean"][15]
                    z[1,28] = importances["importances_mean"][16]
                    z[1,29] = importances["importances_mean"][17]
                    z[1,30] = importances["importances_mean"][18]
                    z[1,31] = importances["importances_std"][1]
                    z[1,32] = importances["importances_std"][2]
                    z[1,33] = importances["importances_std"][3]
                    z[1,34] = importances["importances_std"][4]
                    z[1,35] = importances["importances_std"][5]
                    z[1,36] = importances["importances_std"][6]
                    z[1,37] = importances["importances_std"][7]
                    z[1,38] = importances["importances_std"][8]
                    z[1,39] = importances["importances_std"][9]
                    z[1,40] = importances["importances_std"][10]
                    z[1,41] = importances["importances_std"][11]
                    z[1,42] = importances["importances_std"][12]
                    z[1,43] = importances["importances_std"][13]
                    z[1,44] = importances["importances_std"][14]
                    z[1,45] = importances["importances_std"][15]
                    z[1,46] = importances["importances_std"][16]
                    z[1,47] = importances["importances_std"][17]
                    z[1,48] = importances["importances_std"][18]
                else
                    itrain = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    jtrain = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    ival = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                    jval = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    traincvtrain = avgScore(f1_10_train, 3) 
                    f1s = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    mccs = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    rec = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                    im1 = importances["importances_mean"][1]
                    im2 = importances["importances_mean"][2]
                    im3 = importances["importances_mean"][3]
                    im4 = importances["importances_mean"][4]
                    im5 = importances["importances_mean"][5]
                    im6 = importances["importances_mean"][6]
                    im7 = importances["importances_mean"][7]
                    im8 = importances["importances_mean"][8]
                    im9 = importances["importances_mean"][9]
                    im10 = importances["importances_mean"][10]
                    im11 = importances["importances_mean"][11]
                    im12 = importances["importances_mean"][12]
                    im13 = importances["importances_mean"][13]
                    im14 = importances["importances_mean"][14]
                    im15 = importances["importances_mean"][15]
                    im16 = importances["importances_mean"][16]
                    im17 = importances["importances_mean"][17]
                    im18 = importances["importances_mean"][18]
                    sd1 = importances["importances_std"][1]
                    sd2 = importances["importances_std"][2]
                    sd3 = importances["importances_std"][3]
                    sd4 = importances["importances_std"][4]
                    sd5 = importances["importances_std"][5]
                    sd6 = importances["importances_std"][6]
                    sd7 = importances["importances_std"][7]
                    sd8 = importances["importances_std"][8]
                    sd9 = importances["importances_std"][9]
                    sd10 = importances["importances_std"][10]
                    sd11 = importances["importances_std"][11]
                    sd12 = importances["importances_std"][12]
                    sd13 = importances["importances_std"][13]
                    sd14 = importances["importances_std"][14]
                    sd15 = importances["importances_std"][15]
                    sd16 = importances["importances_std"][16]
                    sd17 = importances["importances_std"][17]
                    sd18 = importances["importances_std"][18]
                    z = vcat(z, [pnn s c itrain jtrain ival jval traincvtrain f1s mccs rec rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 im11 im12 im13 im14 im15 im16 im17 im18 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10 sd11 sd12 sd13 sd14 sd15 sd16 sd17 sd18])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(Penalty = z[:,1], Solver = z[:,2], C_value = z[:,3], f1_train = z[:,4], mcc_train = z[:,5], f1_ext = z[:,6], mcc_ext = z[:,7], f1_3Ftrain = z[:,8], f1_fna = z[:,9], mcc_fna = z[:,10], recall = z[:,11], state = z[:,12], im1 = z[:,13], im2 = z[:,14], im3 = z[:,15], im4 = z[:,16], im5 = z[:,17], im6 = z[:,18], im7 = z[:,19], im8 = z[:,20], im9 = z[:,21], im10 = z[:,22], im11 = z[:,23], im12 = z[:,24], im13 = z[:,25], im14 = z[:,26], im15 = z[:,27], im16 = z[:,28], im17 = z[:,29], im18 = z[:,30], sd1 = z[:,31], sd2 = z[:,32], sd3 = z[:,33], sd4 = z[:,34], sd5 = z[:,35], sd6 = z[:,36], sd7 = z[:,37], sd8 = z[:,38], sd9 = z[:,39], sd10 = z[:,40], sd11 = z[:,41], sd12 = z[:,42], sd13 = z[:,43], sd14 = z[:,44], sd15 = z[:,45], sd16 = z[:,46], sd17 = z[:,47], sd18 = z[:,48])
    z_df_sorted = sort(z_df, [:recall, :f1_fna, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

## call Logistic Regression ##
optiSearch_df = optimLR(trainDEFSDf, ingestedDEFSDf, extDEFSDf, fnaDEFSDf)

## save ##
savePath = "G:\\raMSIn\\XGB_Importance2\\hyperparameterTuning_modelSelection_LR3.csv"
CSV.write(savePath, optiSearch_df)


# ==================================================================================================
## define a function for Support Vector Machine ##
function optimSVM(inputDB, inputDB_ingested, inputDB_ext, inputDB_FNA)

    penalty_r = ["l1", "l2"]  # 2
    loss_r = ["hinge", "squared_hinge"]  # 2
    #gamma_r = ["scale", "auto"] # 2
    #kernel_r = ["linear", "poly", "rbf", "sigmoid"]  # 4
    c_values_r = vcat(10, 5, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001)  # 15
    c_values_r = vcat(0.00009, 0.00008, 0.00007, 0.00006, 0.00005, 0.00004, 0.00003, 0.00002, 0.00001, 0.000009, 0.000008, 0.000007, 0.000006, 0.000005, 0.000004, 0.000003, 0.000002, 0.000001, 0.0000009, 0.0000008, 0.0000007, 0.0000006, 0.0000005, 0.0000004, 0.0000003, 0.0000002, 0.0000001)  # 15
    
    rs = 42
    z = zeros(1,48)
    itr = 1

    N_train = inputDB
    M_train = inputDB_ingested
    M_ext = inputDB_ext
    M_FNA = inputDB_FNA

    for p in 1:2
        if (p == 1)
            continue
        elseif (p == 2)
            for l in 1:2
                for c in c_values_r
                    println("itr=", itr, ", P=", p, ", L=", l, ", C=", c)
                    println("## loading in data ##")
                    Xx_train = deepcopy(M_train[:, 2:end-1])
                    nn_train = deepcopy(N_train[:, 2:end-1])
                    Xx_Ext = deepcopy(M_ext[:, 2:end-1])
                    Xx_FNA = deepcopy(M_FNA[:, 2:end-1])
                    #
                    Yy_train = deepcopy(M_train[:, end])
                    mm_train = deepcopy(N_train[:, end])
                    Yy_Ext = deepcopy(M_ext[:, end])
                    Yy_FNA = deepcopy(M_FNA[:, end])
                    println("## Classification ##")
                    reg = LinearSVC(penalty=penalty_r[p], loss=loss_r[l], C=c, random_state=rs, class_weight=Dict(0=>0.9628, 1=>1.0402))
                    println("## fit ##")
                    fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                    importances = permutation_importance(reg, Matrix(Xx_FNA), Vector(Yy_FNA), n_repeats=10, random_state=42)
                    print(importances["importances_mean"])
                    if itr == 1
                        z[1,1] = p
                        z[1,2] = l
                        z[1,3] = c
                        z[1,4] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                        z[1,5] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                        z[1,6] = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                        z[1,7] = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                        println("## CV ##")
                        f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                        z[1,8] = avgScore(f1_10_train, 3)
                        z[1,9] = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        z[1,10] = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        z[1,11] = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                        z[1,12] = rs
                        z[1,13] = importances["importances_mean"][1]
                        z[1,14] = importances["importances_mean"][2]
                        z[1,15] = importances["importances_mean"][3]
                        z[1,16] = importances["importances_mean"][4]
                        z[1,17] = importances["importances_mean"][5]
                        z[1,18] = importances["importances_mean"][6]
                        z[1,19] = importances["importances_mean"][7]
                        z[1,20] = importances["importances_mean"][8]
                        z[1,21] = importances["importances_mean"][9]
                        z[1,22] = importances["importances_mean"][10]
                        z[1,23] = importances["importances_mean"][11]
                        z[1,24] = importances["importances_mean"][12]
                        z[1,25] = importances["importances_mean"][13]
                        z[1,26] = importances["importances_mean"][14]
                        z[1,27] = importances["importances_mean"][15]
                        z[1,28] = importances["importances_mean"][16]
                        z[1,29] = importances["importances_mean"][17]
                        z[1,30] = importances["importances_mean"][18]
                        z[1,31] = importances["importances_std"][1]
                        z[1,32] = importances["importances_std"][2]
                        z[1,33] = importances["importances_std"][3]
                        z[1,34] = importances["importances_std"][4]
                        z[1,35] = importances["importances_std"][5]
                        z[1,36] = importances["importances_std"][6]
                        z[1,37] = importances["importances_std"][7]
                        z[1,38] = importances["importances_std"][8]
                        z[1,39] = importances["importances_std"][9]
                        z[1,40] = importances["importances_std"][10]
                        z[1,41] = importances["importances_std"][11]
                        z[1,42] = importances["importances_std"][12]
                        z[1,43] = importances["importances_std"][13]
                        z[1,44] = importances["importances_std"][14]
                        z[1,45] = importances["importances_std"][15]
                        z[1,46] = importances["importances_std"][16]
                        z[1,47] = importances["importances_std"][17]
                        z[1,48] = importances["importances_std"][18]
                    else
                        itrain = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                        jtrain = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                        ival = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                        jval = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                        println("## CV ##")
                        f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                        traincvtrain = avgScore(f1_10_train, 3) 
                        f1s = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        mccs = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        rec = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                        im1 = importances["importances_mean"][1]
                        im2 = importances["importances_mean"][2]
                        im3 = importances["importances_mean"][3]
                        im4 = importances["importances_mean"][4]
                        im5 = importances["importances_mean"][5]
                        im6 = importances["importances_mean"][6]
                        im7 = importances["importances_mean"][7]
                        im8 = importances["importances_mean"][8]
                        im9 = importances["importances_mean"][9]
                        im10 = importances["importances_mean"][10]
                        im11 = importances["importances_mean"][11]
                        im12 = importances["importances_mean"][12]
                        im13 = importances["importances_mean"][13]
                        im14 = importances["importances_mean"][14]
                        im15 = importances["importances_mean"][15]
                        im16 = importances["importances_mean"][16]
                        im17 = importances["importances_mean"][17]
                        im18 = importances["importances_mean"][18]
                        sd1 = importances["importances_std"][1]
                        sd2 = importances["importances_std"][2]
                        sd3 = importances["importances_std"][3]
                        sd4 = importances["importances_std"][4]
                        sd5 = importances["importances_std"][5]
                        sd6 = importances["importances_std"][6]
                        sd7 = importances["importances_std"][7]
                        sd8 = importances["importances_std"][8]
                        sd9 = importances["importances_std"][9]
                        sd10 = importances["importances_std"][10]
                        sd11 = importances["importances_std"][11]
                        sd12 = importances["importances_std"][12]
                        sd13 = importances["importances_std"][13]
                        sd14 = importances["importances_std"][14]
                        sd15 = importances["importances_std"][15]
                        sd16 = importances["importances_std"][16]
                        sd17 = importances["importances_std"][17]
                        sd18 = importances["importances_std"][18]
                        z = vcat(z, [p l c itrain jtrain ival jval traincvtrain f1s mccs rec rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 im11 im12 im13 im14 im15 im16 im17 im18 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10 sd11 sd12 sd13 sd14 sd15 sd16 sd17 sd18])
                        println(z[end, :])
                    end
                    println("End of ", itr, " iterations")
                    itr += 1
                end
            end
        end
        z_df = DataFrame(penalty = z[:,1], loss = z[:,2], C_value = z[:,3], f1_train = z[:,4], mcc_train = z[:,5], f1_ext = z[:,6], mcc_ext = z[:,7], f1_3Ftrain = z[:,8], f1_fna = z[:,9], mcc_fna = z[:,10], recall = z[:,11], state = z[:,12], im1 = z[:,13], im2 = z[:,14], im3 = z[:,15], im4 = z[:,16], im5 = z[:,17], im6 = z[:,18], im7 = z[:,19], im8 = z[:,20], im9 = z[:,21], im10 = z[:,22], im11 = z[:,23], im12 = z[:,24], im13 = z[:,25], im14 = z[:,26], im15 = z[:,27], im16 = z[:,28], im17 = z[:,29], im18 = z[:,30], sd1 = z[:,31], sd2 = z[:,32], sd3 = z[:,33], sd4 = z[:,34], sd5 = z[:,35], sd6 = z[:,36], sd7 = z[:,37], sd8 = z[:,38], sd9 = z[:,39], sd10 = z[:,40], sd11 = z[:,41], sd12 = z[:,42], sd13 = z[:,43], sd14 = z[:,44], sd15 = z[:,45], sd16 = z[:,46], sd17 = z[:,47], sd18 = z[:,48])
        z_df_sorted = sort(z_df, [:recall, :f1_fna, :f1_3Ftrain], rev=true)
        return z_df_sorted
    end
end

## call Support Vector Machine ##
optiSearch_df = optimSVM(trainDEFSDf, ingestedDEFSDf, extDEFSDf, fnaDEFSDf)

## save ##
savePath = "G:\\raMSIn\\XGB_Importance2\\hyperparameterTuning_modelSelection_SVM2.csv"
CSV.write(savePath, optiSearch_df)


# ==================================================================================================
## define a function for Gradient Boost ##
function optimGradientBoostClass(inputDB, inputDB_ingested, inputDB_ext, inputDB_FNA)
    #lr_r = vcat(0.3, 0.1)  # 2
    lr_r = vcat(0.3, 0.2, 0.1)  # 3
    #leaf_r = vcat(8, 12, 18)  # 3
    leaf_r = vcat(18, 24)  # 2
    #depth_r = vcat(collect(6:2:10))  # 3
    depth_r = vcat(collect(8:2:12))  # 3
    #split_r = vcat(collect(10:10:20))  # 2
    #split_r = vcat(collect(10:10:20))  # 2
    split_r = vcat(collect(10:5:25))  # 4
    #tree_r = vcat(collect(50:100:250))  # 3
    tree_r = vcat(collect(25:25:75))  # 3
    
    rs = 42
    z = zeros(1,50)
    itr = 1

    N_train = inputDB
    M_train = inputDB_ingested
    M_ext = inputDB_ext
    M_FNA = inputDB_FNA

    for lr in lr_r
        for l in leaf_r
            for d in depth_r
                for s in split_r
                    for t in tree_r
                        println("itr=", itr, ", lr=", lr, ", leaf=", l, ", depth=", d, ", minSsplit=", s, ", tree=", t)
                        println("## loading in data ##")
                        Xx_train = deepcopy(M_train[:, 2:end-1])
                        nn_train = deepcopy(N_train[:, 2:end-1])
                        Xx_Ext = deepcopy(M_ext[:, 2:end-1])
                        Xx_FNA = deepcopy(M_FNA[:, 2:end-1])
                        #
                        Yy_train = deepcopy(M_train[:, end])
                        mm_train = deepcopy(N_train[:, end])
                        Yy_Ext = deepcopy(M_ext[:, end])
                        Yy_FNA = deepcopy(M_FNA[:, end])
                        println("## Classification ##")
                        reg = GradientBoostingClassifier(learning_rate=lr, n_estimators=t, max_depth=d, min_samples_leaf=l, min_samples_split=s, random_state=rs, n_iter_no_change=5)
                        println("## fit ##")
                        fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                        importances = permutation_importance(reg, Matrix(Xx_FNA), Vector(Yy_FNA), n_repeats=10, random_state=42, n_jobs=-1)
                        print(importances["importances_mean"])
                        if itr == 1
                            z[1,1] = lr
                            z[1,2] = l
                            z[1,3] = t
                            z[1,4] = d
                            z[1,5] = s
                            z[1,6] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                            z[1,7] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                            z[1,8] = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                            z[1,9] = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                            println("## CV ##")
                            f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                            z[1,10] = avgScore(f1_10_train, 3)
                            z[1,11] = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                            z[1,12] = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                            z[1,13] = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                            z[1,14] = rs
                            z[1,15] = importances["importances_mean"][1]
                            z[1,16] = importances["importances_mean"][2]
                            z[1,17] = importances["importances_mean"][3]
                            z[1,18] = importances["importances_mean"][4]
                            z[1,19] = importances["importances_mean"][5]
                            z[1,20] = importances["importances_mean"][6]
                            z[1,21] = importances["importances_mean"][7]
                            z[1,22] = importances["importances_mean"][8]
                            z[1,23] = importances["importances_mean"][9]
                            z[1,24] = importances["importances_mean"][10]
                            z[1,25] = importances["importances_mean"][11]
                            z[1,26] = importances["importances_mean"][12]
                            z[1,27] = importances["importances_mean"][13]
                            z[1,28] = importances["importances_mean"][14]
                            z[1,29] = importances["importances_mean"][15]
                            z[1,30] = importances["importances_mean"][16]
                            z[1,31] = importances["importances_mean"][17]
                            z[1,32] = importances["importances_mean"][18]
                            z[1,33] = importances["importances_std"][1]
                            z[1,34] = importances["importances_std"][2]
                            z[1,35] = importances["importances_std"][3]
                            z[1,36] = importances["importances_std"][4]
                            z[1,37] = importances["importances_std"][5]
                            z[1,38] = importances["importances_std"][6]
                            z[1,39] = importances["importances_std"][7]
                            z[1,40] = importances["importances_std"][8]
                            z[1,41] = importances["importances_std"][9]
                            z[1,42] = importances["importances_std"][10]
                            z[1,43] = importances["importances_std"][11]
                            z[1,44] = importances["importances_std"][12]
                            z[1,45] = importances["importances_std"][13]
                            z[1,46] = importances["importances_std"][14]
                            z[1,47] = importances["importances_std"][15]
                            z[1,48] = importances["importances_std"][16]
                            z[1,49] = importances["importances_std"][17]
                            z[1,50] = importances["importances_std"][18]
                        else
                            itrain = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                            jtrain = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                            ival = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                            jval = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                            println("## CV ##")
                            f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                            traincvtrain = avgScore(f1_10_train, 3) 
                            f1s = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                            mccs = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                            rec = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                            im1 = importances["importances_mean"][1]
                            im2 = importances["importances_mean"][2]
                            im3 = importances["importances_mean"][3]
                            im4 = importances["importances_mean"][4]
                            im5 = importances["importances_mean"][5]
                            im6 = importances["importances_mean"][6]
                            im7 = importances["importances_mean"][7]
                            im8 = importances["importances_mean"][8]
                            im9 = importances["importances_mean"][9]
                            im10 = importances["importances_mean"][10]
                            im11 = importances["importances_mean"][11]
                            im12 = importances["importances_mean"][12]
                            im13 = importances["importances_mean"][13]
                            im14 = importances["importances_mean"][14]
                            im15 = importances["importances_mean"][15]
                            im16 = importances["importances_mean"][16]
                            im17 = importances["importances_mean"][17]
                            im18 = importances["importances_mean"][18]
                            sd1 = importances["importances_std"][1]
                            sd2 = importances["importances_std"][2]
                            sd3 = importances["importances_std"][3]
                            sd4 = importances["importances_std"][4]
                            sd5 = importances["importances_std"][5]
                            sd6 = importances["importances_std"][6]
                            sd7 = importances["importances_std"][7]
                            sd8 = importances["importances_std"][8]
                            sd9 = importances["importances_std"][9]
                            sd10 = importances["importances_std"][10]
                            sd11 = importances["importances_std"][11]
                            sd12 = importances["importances_std"][12]
                            sd13 = importances["importances_std"][13]
                            sd14 = importances["importances_std"][14]
                            sd15 = importances["importances_std"][15]
                            sd16 = importances["importances_std"][16]
                            sd17 = importances["importances_std"][17]
                            sd18 = importances["importances_std"][18]
                            z = vcat(z, [lr l t d s itrain jtrain ival jval traincvtrain f1s mccs rec rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 im11 im12 im13 im14 im15 im16 im17 im18 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10 sd11 sd12 sd13 sd14 sd15 sd16 sd17 sd18])
                            println(z[end, :])
                        end
                        println("End of ", itr, " iterations")
                        itr += 1
                    end
                end
            end
        end
    end
    z_df = DataFrame(lr = z[:,1], leaves = z[:,2], trees = z[:,3], depth = z[:,4], minSplit = z[:,5], f1_train = z[:,6], mcc_train = z[:,7], f1_ext = z[:,8], mcc_ext = z[:,9], f1_3Ftrain = z[:,10], f1_fna = z[:,11], mcc_fna = z[:,12], recall = z[:,13], state = z[:,14], im1 = z[:,15], im2 = z[:,16], im3 = z[:,17], im4 = z[:,18], im5 = z[:,19], im6 = z[:,20], im7 = z[:,21], im8 = z[:,22], im9 = z[:,23], im10 = z[:,24], im11 = z[:,25], im12 = z[:,26], im13 = z[:,27], im14 = z[:,28], im15 = z[:,29], im16 = z[:,30], im17 = z[:,31], im18 = z[:,32], sd1 = z[:,33], sd2 = z[:,34], sd3 = z[:,35], sd4 = z[:,36], sd5 = z[:,37], sd6 = z[:,38], sd7 = z[:,39], sd8 = z[:,40], sd9 = z[:,41], sd10 = z[:,42], sd11 = z[:,43], sd12 = z[:,44], sd13 = z[:,45], sd14 = z[:,46], sd15 = z[:,47], sd16 = z[:,48], sd17 = z[:,49], sd18 = z[:,50])
    z_df_sorted = sort(z_df, [:recall, :f1_fna, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

## call Gradient Boost ##
optiSearch_df = optimGradientBoostClass(trainDEFSDf, ingestedDEFSDf, extDEFSDf, fnaDEFSDf)

## save ##
savePath = "G:\\raMSIn\\XGB_Importance2\\hyperparameterTuning_modelSelection_GBM2.csv"
CSV.write(savePath, optiSearch_df)


# ==================================================================================================
## define a function for Ada Boost ##
function optimAdaBoostClass(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    lr_r = vcat(1, 0.1, 0.01, 0.001)  # 4
    leaf_r = vcat(1,2)
    tree_r = vcat(collect(50:100:350))  # 4
    depth_r = vcat(collect(2:1:6))  # 5
    split_r = vcat(2)
    rs = 42
    z = zeros(1,31)
    l = 2
    r = 2
    mod = 0
    rank = vcat(5,6,7,9,10,13,14, 17)
    N_train = inputDB
    M_train = vcat(inputDB, inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :])
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    itr = 1
    for lr in lr_r
        for t in tree_r
            for d in depth_r
                dtc = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, min_samples_split=r, random_state=rs, class_weight=Dict(0=>0.9929, 1=>1.0072))
                println("itr=", itr, ", lr=", lr, ", tree=", t, ", model=", mod)
                println("## loading in data ##")
                Xx_train = deepcopy(M_train[:, rank])
                nn_train = deepcopy(N_train[:, rank])
                Xx_val = deepcopy(M_val[:, rank])
                Xx_test = deepcopy(M_pest[:, rank])
                Xx_test2 = deepcopy(M_pest2[:, rank])
                #
                Yy_train = deepcopy(M_train[:, end-4])
                mm_train = deepcopy(N_train[:, end-4])
                Yy_val = deepcopy(M_val[:, end-4])
                Yy_test = deepcopy(M_pest[:, end-1])
                Yy_test2 = deepcopy(M_pest2[:, end-1])
                println("## Classification ##")
                reg = AdaBoostClassifier(estimator=dtc, n_estimators=t, algorithm="SAMME", learning_rate=lr, random_state=rs)
                println("## fit ##")
                fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42)
                print(importances["importances_mean"])
                if itr == 1
                    z[1,1] = lr
                    z[1,2] = l
                    z[1,3] = t
                    z[1,4] = d
                    z[1,5] = r
                    z[1,6] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,7] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    z[1,8] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    z[1,9] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    z[1,10] = avgScore(f1_10_train, 3)
                    z[1,11] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,12] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    z[1,13] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                    z[1,14] = rs
                    z[1,15] = mod
                    z[1,16] = importances["importances_mean"][1]
                    z[1,17] = importances["importances_mean"][2]
                    z[1,18] = importances["importances_mean"][3]
                    z[1,19] = importances["importances_mean"][4]
                    z[1,20] = importances["importances_mean"][5]
                    z[1,21] = importances["importances_mean"][6]
                    z[1,22] = importances["importances_mean"][7]
                    z[1,23] = importances["importances_mean"][8]
                    z[1,24] = importances["importances_std"][1]
                    z[1,25] = importances["importances_std"][2]
                    z[1,26] = importances["importances_std"][3]
                    z[1,27] = importances["importances_std"][4]
                    z[1,28] = importances["importances_std"][5]
                    z[1,29] = importances["importances_std"][6]
                    z[1,30] = importances["importances_std"][7]
                    z[1,31] = importances["importances_std"][8]
                    println(z[end, :])
                else
                    itrain = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    jtrain = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                    println("## CV ##")
                    f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                    traincvtrain = avgScore(f1_10_train, 3) 
                    f1s = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    mccs = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                    rec = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                    im1 = importances["importances_mean"][1]
                    im2 = importances["importances_mean"][2]
                    im3 = importances["importances_mean"][3]
                    im4 = importances["importances_mean"][4]
                    im5 = importances["importances_mean"][5]
                    im6 = importances["importances_mean"][6]
                    im7 = importances["importances_mean"][7]
                    im8 = importances["importances_mean"][8]
                    sd1 = importances["importances_std"][1]
                    sd2 = importances["importances_std"][2]
                    sd3 = importances["importances_std"][3]
                    sd4 = importances["importances_std"][4]
                    sd5 = importances["importances_std"][5]
                    sd6 = importances["importances_std"][6]
                    sd7 = importances["importances_std"][7]
                    sd8 = importances["importances_std"][8]
                    z = vcat(z, [lr l t d r itrain jtrain ival jval traincvtrain f1s mccs rec rs mod im1 im2 im3 im4 im5 im6 im7 im8 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(learnRate = z[:,1], leaves = z[:,2], trees = z[:,3], depth = z[:,4], minSplit = z[:,5], f1_train = z[:,6], mcc_train = z[:,7], f1_val = z[:,8], mcc_val = z[:,9], f1_3Ftrain = z[:,10], f1_pest = z[:,11], mcc_pest = z[:,12], recall = z[:,13], state = z[:,14], model = z[:,15], im1 = z[:,16], im2 = z[:,17], im3 = z[:,18], im4 = z[:,19], im5 = z[:,20], im6 = z[:,21], im7 = z[:,22], im8 = z[:,23], sd1 = z[:,24], sd2 = z[:,25], sd3 = z[:,26], sd4 = z[:,27], sd5 = z[:,28], sd6 = z[:,29], sd7 = z[:,30], sd8 = z[:,31])
    z_df_sorted = sort(z_df, [:recall, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

## call Ada Boost ##
optiSearch_df = optimAdaBoostClass(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)

## save ##
savePath = "F:\\UvA\\app\\hyperparameterTuning_modelSelection_Ada(0)_noFilter.csv"
CSV.write(savePath, optiSearch_df)


# ==================================================================================================
## define a function for Multi-Layer Perceptrons ##
function optimMLP(inputDB, inputDB_test, inputDB_pest, inputDB_pest2)
    hls_r = [(8,8,8), (8,16,8), (8,16,16), (16,16,16), (16,16,8), (16,8,8), (16,8,16), (8,16,16)]  # 8
    maxIter_r = vcat(100, 200)  # 2
    alpha_r = [0.0001, 0.05]  # 2
    act_r = ["tanh", "relu"]  # 2
    solver_r = ["sgd", "adam"]  # 2
    lr_r = ["constant", "adaptive"]  # 2
    rs = 42
    z = zeros(1,32)
    mod = 0
    rank = vcat(5,6,7,9,10,13,14, 17)
    N_train = inputDB
    M_train = vcat(inputDB, inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :], inputDB[inputDB.LABEL .== 1, :])
    M_val = inputDB_test
    M_pest = inputDB_pest
    M_pest2 = inputDB_pest2
    itr = 1
    for hls in 1:8
        for it in maxIter_r
            for alph in alpha_r
                for act in vcat(1,2)
                    for sol in vcat(1,2)
                        for lr in vcat(1,2)
                            println("itr=", itr, ", hls=", hls, ", maxit=", it, ", act=", act, ", solver=", sol, ", alph=", alph, ", lr=", lr, ", model=", mod)
                            println("## loading in data ##")
                            Xx_train = deepcopy(M_train[:, rank])
                            nn_train = deepcopy(N_train[:, rank])
                            Xx_val = deepcopy(M_val[:, rank])
                            Xx_test = deepcopy(M_pest[:, rank])
                            Xx_test2 = deepcopy(M_pest2[:, rank])
                            #
                            Yy_train = deepcopy(M_train[:, end-4])
                            mm_train = deepcopy(N_train[:, end-4])
                            Yy_val = deepcopy(M_val[:, end-4])
                            Yy_test = deepcopy(M_pest[:, end-1])
                            Yy_test2 = deepcopy(M_pest2[:, end-1])
                            println("## Classification ##")
                            reg = MLPClassifier(hidden_layer_sizes=hls_r[hls], max_iter=it, activation=act_r[act], solver=solver_r[sol], alpha=alph, learning_rate=lr_r[lr], random_state=rs)  # 0.7263; 1.6048
                            println("## fit ##")
                            fit!(reg, Matrix(Xx_train), Vector(Yy_train))
                            importances = permutation_importance(reg, Matrix(Xx_test), Vector(Yy_test), n_repeats=10, random_state=42)
                            print(importances["importances_mean"])
                            if itr == 1
                                z[1,1] = hls
                                z[1,2] = it
                                z[1,3] = alph
                                z[1,4] = act
                                z[1,5] = sol
                                z[1,6] = lr
                                z[1,7] = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                                z[1,8] = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                                z[1,9] = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                                z[1,10] = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                                println("## CV ##")
                                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                                z[1,11] = avgScore(f1_10_train, 3)
                                z[1,12] = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                                z[1,13] = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                                z[1,14] = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                                z[1,15] = rs
                                z[1,16] = mod
                                z[1,17] = importances["importances_mean"][1]
                                z[1,18] = importances["importances_mean"][2]
                                z[1,19] = importances["importances_mean"][3]
                                z[1,20] = importances["importances_mean"][4]
                                z[1,21] = importances["importances_mean"][5]
                                z[1,22] = importances["importances_mean"][6]
                                z[1,23] = importances["importances_mean"][7]
                                z[1,24] = importances["importances_mean"][8]
                                z[1,25] = importances["importances_std"][1]
                                z[1,26] = importances["importances_std"][2]
                                z[1,27] = importances["importances_std"][3]
                                z[1,28] = importances["importances_std"][4]
                                z[1,29] = importances["importances_std"][5]
                                z[1,30] = importances["importances_std"][6]
                                z[1,31] = importances["importances_std"][7]
                                z[1,32] = importances["importances_std"][8]
                                println(z[end, :])
                            else
                                itrain = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                                jtrain = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                                ival = f1_score(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                                jval = matthews_corrcoef(Vector(Yy_val), predict(reg, Matrix(Xx_val)), sample_weight=sampletestW)
                                println("## CV ##")
                                f1_10_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 3, scoring=f1)
                                traincvtrain = avgScore(f1_10_train, 3) 
                                f1s = f1_score(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                                mccs = matthews_corrcoef(Vector(Yy_test), predict(reg, Matrix(Xx_test)), sample_weight=samplepestW)
                                rec = recall_score(Vector(Yy_test2), predict(reg, Matrix(Xx_test2)))
                                im1 = importances["importances_mean"][1]
                                im2 = importances["importances_mean"][2]
                                im3 = importances["importances_mean"][3]
                                im4 = importances["importances_mean"][4]
                                im5 = importances["importances_mean"][5]
                                im6 = importances["importances_mean"][6]
                                im7 = importances["importances_mean"][7]
                                im8 = importances["importances_mean"][8]
                                sd1 = importances["importances_std"][1]
                                sd2 = importances["importances_std"][2]
                                sd3 = importances["importances_std"][3]
                                sd4 = importances["importances_std"][4]
                                sd5 = importances["importances_std"][5]
                                sd6 = importances["importances_std"][6]
                                sd7 = importances["importances_std"][7]
                                sd8 = importances["importances_std"][8]
                                z = vcat(z, [hls it alph act sol lr itrain jtrain ival jval traincvtrain f1s mccs rec rs mod im1 im2 im3 im4 im5 im6 im7 im8 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8])
                                println(z[end, :])
                            end
                            println("End of ", itr, " iterations")
                            itr += 1
                        end
                    end
                end
            end
        end
    end
    z_df = DataFrame(layers = z[:,1], maxIt = z[:,2], alpha = z[:,3], act = z[:,4], solver = z[:,5], lr = z[:,6], f1_train = z[:,7], mcc_train = z[:,8], f1_val = z[:,9], mcc_val = z[:,10], f1_3Ftrain = z[:,11], f1_pest = z[:,12], mcc_pest = z[:,13], recall = z[:,14], state = z[:,15], model = z[:,16], im1 = z[:,17], im2 = z[:,18], im3 = z[:,19], im4 = z[:,20], im5 = z[:,21], im6 = z[:,22], im7 = z[:,23], im8 = z[:,24], sd1 = z[:,25], sd2 = z[:,26], sd3 = z[:,27], sd4 = z[:,28], sd5 = z[:,29], sd6 = z[:,30], sd7 = z[:,31], sd8 = z[:,32])
    z_df_sorted = sort(z_df, [:recall, :f1_pest, :f1_3Ftrain], rev=true)
    return z_df_sorted
end

## call Multi-Layer Perceptrons ##
optiSearch_df = optimMLP(trainDEFSDf, testDEFSDf, noTeaDEFSDf, TeaDEFSDf)

# save, ouputing 180 x 8 df
savePath = "F:\\UvA\\app\\hyperparameterTuning_modelSelection_MLP(0)_noFilter.csv"
CSV.write(savePath, optiSearch_df)
