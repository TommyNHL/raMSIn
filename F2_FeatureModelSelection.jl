VERSION
## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
import Conda
Conda.PYTHONDIR
ENV["PYTHON"] = raw"C:\Users\T1208\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
#ENV["PYTHON"] = raw"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"  # python 3.11
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
trainDEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\raMSIn\\df_train_raMSIn4nonInDI_norm.csv", DataFrame)
trainDEFSDf = trainDEFSDf[:, vcat(1, collect(2:9), 16, 17, end)]
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
extDEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\raMSIn\\df_ext_raMSIn4nonInDI_norm.csv", DataFrame)
extDEFSDf = extDEFSDf[:, vcat(1, collect(2:9), 16, 17, end)]
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
ingestedDEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\raMSIn\\df_ingested_raMSIn4nonInDI_norm.csv", DataFrame)
ingestedDEFSDf = ingestedDEFSDf[:, vcat(1, collect(2:9), 16, 17, end)]
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
fnaDEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\raMSIn\\df_FNA_raMSIn4nonInDI_norm.csv", DataFrame)
fnaDEFSDf = fnaDEFSDf[:, vcat(1, collect(2:9), 16, 17, end)]
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

## input DirectIn set ## 88701 x 20 df
diDEFSDf = CSV.read("C:\\Users\\T1208\\PyLAB\\raMSIn\\df_nonInDI_raMSIn4nonInDI_norm.csv", DataFrame)
diDEFSDf = diDEFSDf[:, vcat(1, collect(2:9), 16, 17, end)]
diDEFSDf[diDEFSDf.type .== 1, :]
    ## calculate weight ##  0: 3027, 1: 3030
    Yy_DI = deepcopy(diDEFSDf[:, end])  # 1.0005; 0.9995
    sampleDiW = []
    for w in Vector(Yy_DI)
        if w == 0
            push!(sampleDiW, 1.0005)
        elseif w == 1
            push!(sampleDiW, 0.9995)
        end
    end  

## define functions for performace evaluation ##
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
function optimRandomForestClass(inputDB, inputDB_ingested, inputDB_ext, inputDB_FNA, inputDB_di)
    #leaf_r = vcat(2, 4, 8, 12, 18)  # 5
    #leaf_r = vcat(collect(2:1:8))  # 7
    leaf_r = vcat(collect(18:1:28))  # 11
    #depth_r = vcat(collect(2:1:10))  # 9
    #depth_r = vcat(collect(6:1:14))  # 9
    depth_r = vcat(collect(4:1:10))  # 7
    #split_r = vcat(collect(2:1:10))  # 9
    split_r = vcat(collect(10:5:20))  # 3
    tree_r = vcat(collect(50:50:300))  # 6

    rs = 42
    z = zeros(1,36)
    itr = 1

    N_train = inputDB
    M_train = inputDB_ingested
    M_ext = inputDB_ext
    M_FNA = inputDB_FNA
    M_di = inputDB_di

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
                    Xx_di = deepcopy(M_di[:, 2:end-1])
                    #
                    Yy_train = deepcopy(M_train[:, end])
                    mm_train = deepcopy(N_train[:, end])
                    Yy_Ext = deepcopy(M_ext[:, end])
                    Yy_FNA = deepcopy(M_FNA[:, end])
                    Yy_di = deepcopy(M_di[:, end])
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
                        f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                        z[1,9] = avgScore(f1_5_train, 5)
                        z[1,10] = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        z[1,11] = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        z[1,12] = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                        z[1,13] = f1_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                        z[1,14] = matthews_corrcoef(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                        z[1,15] = recall_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)))
                        z[1,16] = rs
                        z[1,17] = importances["importances_mean"][1]
                        z[1,18] = importances["importances_mean"][2]
                        z[1,19] = importances["importances_mean"][3]
                        z[1,20] = importances["importances_mean"][4]
                        z[1,21] = importances["importances_mean"][5]
                        z[1,22] = importances["importances_mean"][6]
                        z[1,23] = importances["importances_mean"][7]
                        z[1,24] = importances["importances_mean"][8]
                        z[1,25] = importances["importances_mean"][9]
                        z[1,26] = importances["importances_mean"][10]
                        z[1,27] = importances["importances_std"][1]
                        z[1,28] = importances["importances_std"][2]
                        z[1,29] = importances["importances_std"][3]
                        z[1,30] = importances["importances_std"][4]
                        z[1,31] = importances["importances_std"][5]
                        z[1,32] = importances["importances_std"][6]
                        z[1,33] = importances["importances_std"][7]
                        z[1,34] = importances["importances_std"][8]
                        z[1,35] = importances["importances_std"][9]
                        z[1,36] = importances["importances_std"][10]
                    else
                        itrain = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                        jtrain = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                        ival = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                        jval = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                        println("## CV ##")
                        f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                        traincvtrain = avgScore(f1_5_train, 5) 
                        f1s = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        mccs = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        rec = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                        f1s2 = f1_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                        mccs2 = matthews_corrcoef(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                        rec2 = recall_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)))
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
                        z = vcat(z, [l t d s itrain jtrain ival jval traincvtrain f1s mccs rec f1s2 mccs2 rec2 rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10])
                        println(z[end, :])
                    end
                    println("End of ", itr, " iterations")
                    itr += 1
                end
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], depth = z[:,3], minSplit = z[:,4], f1_train = z[:,5], mcc_train = z[:,6], f1_ext = z[:,7], mcc_ext = z[:,8], f1_5Ftrain = z[:,9], f1_fna = z[:,10], mcc_fna = z[:,11], recall_fna = z[:,12], f1_di = z[:,13], mcc_di = z[:,14], recall_di = z[:,15]; state = z[:,16], im1 = z[:,17], im2 = z[:,18], im3 = z[:,19], im4 = z[:,20], im5 = z[:,21], im6 = z[:,22], im7 = z[:,23], im8 = z[:,24], im9 = z[:,25], im10 = z[:,26], sd1 = z[:,27], sd2 = z[:,28], sd3 = z[:,29], sd4 = z[:,30], sd5 = z[:,31], sd6 = z[:,32], sd7 = z[:,33], sd8 = z[:,34], sd9 = z[:,35], sd10 = z[:,36])
    z_df_sorted = sort(z_df, [:recall_di, :recall_fna, :f1_5Ftrain], rev=true)
    return z_df_sorted
end

## call Random Forest ##
optiSearch_df = optimRandomForestClass(trainDEFSDf, ingestedDEFSDf, extDEFSDf, fnaDEFSDf, diDEFSDf)

## save ##
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\modeling\\hyperparameterTuning_modelSelection_RF3.csv"
CSV.write(savePath, optiSearch_df)


# ==================================================================================================
## define a function for Decision Tree ##
function optimDecisionTreeClass(inputDB, inputDB_ingested, inputDB_ext, inputDB_FNA, inputDB_di)
    #leaf_r = vcat(2, 4, 8, 12, 18)  # 5
    #leaf_r = vcat(collect(2:2:12), collect(16:4:48), collect(56:8:80), 100, 200, 500)  # 6+9+4+3=22
    leaf_r = vcat(collect(20:1:30))  # 11
    #depth_r = vcat(collect(2:1:10))  # 9
    #depth_r = vcat(collect(2:1:14))  # 13
    depth_r = vcat(collect(4:1:8))  # 5
    #split_r = vcat(collect(2:1:10))  # 9
    split_r = vcat(collect(2:2:10))  # 5

    rs = 42
    z = zeros(1,35)
    itr = 1

    N_train = inputDB
    M_train = inputDB_ingested
    M_ext = inputDB_ext
    M_FNA = inputDB_FNA
    M_di = inputDB_di

    for l in leaf_r
        for d in depth_r
            for s in split_r
                println("itr=", itr, ", leaf=", l, ", depth=", d, ", minSsplit=", s)
                println("## loading in data ##")
                Xx_train = deepcopy(M_train[:, 2:end-1])
                nn_train = deepcopy(N_train[:, 2:end-1])
                Xx_Ext = deepcopy(M_ext[:, 2:end-1])
                Xx_FNA = deepcopy(M_FNA[:, 2:end-1])
                Xx_di = deepcopy(M_di[:, 2:end-1])
                #
                Yy_train = deepcopy(M_train[:, end])
                mm_train = deepcopy(N_train[:, end])
                Yy_Ext = deepcopy(M_ext[:, end])
                Yy_FNA = deepcopy(M_FNA[:, end])
                Yy_di = deepcopy(M_di[:, end])
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
                    f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                    z[1,8] = avgScore(f1_5_train, 5)
                    z[1,9] = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    z[1,10] = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    z[1,11] = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                    z[1,12] = f1_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                    z[1,13] = matthews_corrcoef(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                    z[1,14] = recall_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)))
                    z[1,15] = rs
                    z[1,16] = importances["importances_mean"][1]
                    z[1,17] = importances["importances_mean"][2]
                    z[1,18] = importances["importances_mean"][3]
                    z[1,19] = importances["importances_mean"][4]
                    z[1,20] = importances["importances_mean"][5]
                    z[1,21] = importances["importances_mean"][6]
                    z[1,22] = importances["importances_mean"][7]
                    z[1,23] = importances["importances_mean"][8]
                    z[1,24] = importances["importances_mean"][9]
                    z[1,25] = importances["importances_mean"][10]
                    z[1,26] = importances["importances_std"][1]
                    z[1,27] = importances["importances_std"][2]
                    z[1,28] = importances["importances_std"][3]
                    z[1,29] = importances["importances_std"][4]
                    z[1,30] = importances["importances_std"][5]
                    z[1,31] = importances["importances_std"][6]
                    z[1,32] = importances["importances_std"][7]
                    z[1,33] = importances["importances_std"][8]
                    z[1,34] = importances["importances_std"][9]
                    z[1,35] = importances["importances_std"][10]
                else
                    itrain = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    jtrain = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    ival = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                    jval = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                    println("## CV ##")
                    f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                    traincvtrain = avgScore(f1_5_train, 5) 
                    f1s = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    mccs = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    rec = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                    f1s2 = f1_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                    mccs2 = matthews_corrcoef(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                    rec2 = recall_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)))
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
                    z = vcat(z, [l d s itrain jtrain ival jval traincvtrain f1s mccs rec f1s2 mccs2 rec2 rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(leaves = z[:,1], depth = z[:,2], minSplit = z[:,3], f1_train = z[:,4], mcc_train = z[:,5], f1_ext = z[:,6], mcc_ext = z[:,7], f1_5Ftrain = z[:,8], f1_fna = z[:,9], mcc_fna = z[:,10], recall_fna = z[:,11], f1_di = z[:,12], mcc_di = z[:,13], recall_di = z[:,14], state = z[:,15], im1 = z[:,16], im2 = z[:,17], im3 = z[:,18], im4 = z[:,19], im5 = z[:,20], im6 = z[:,21], im7 = z[:,22], im8 = z[:,23], im9 = z[:,24], im10 = z[:,25], sd1 = z[:,26], sd2 = z[:,27], sd3 = z[:,28], sd4 = z[:,29], sd5 = z[:,30], sd6 = z[:,31], sd7 = z[:,32], sd8 = z[:,33], sd9 = z[:,34], sd10 = z[:,35])
    z_df_sorted = sort(z_df, [:recall_di, :recall_fna, :f1_5Ftrain], rev=true)
    return z_df_sorted
end

## call Decision Tree ##
optiSearch_df = optimDecisionTreeClass(trainDEFSDf, ingestedDEFSDf, extDEFSDf, fnaDEFSDf, diDEFSDf)

## save ##
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\modeling\\hyperparameterTuning_modelSelection_DT3.csv"
CSV.write(savePath, optiSearch_df)


# ==================================================================================================
## define a function for Logistic Regression ##
function optimLR(inputDB, inputDB_ingested, inputDB_ext, inputDB_FNA, inputDB_di)
    penalty_r = ["l1", "l2"]  # 2
    solver_rs = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]  # 5
    #c_values_r = vcat(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01)  # 14
    #c_values_r = vcat(collect(0.01:0.01:0.05), collect(0.001:0.001:0.009), 0.00075, 0.0005)  # 8
    c_values_r = vcat(collect(0.001:0.00025:0.00375), collect(0.0001:0.00005:0.00095), 0.00001, 0.000025, 0.00005, 0.000075)  # 13

    rs = 42
    z = zeros(1,35)
    itr = 1
    pnn = 0

    N_train = inputDB
    M_train = inputDB_ingested
    M_ext = inputDB_ext
    M_FNA = inputDB_FNA
    M_di = inputDB_di

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
                Xx_di = deepcopy(M_di[:, 2:end-1])
                #
                Yy_train = deepcopy(M_train[:, end])
                mm_train = deepcopy(N_train[:, end])
                Yy_Ext = deepcopy(M_ext[:, end])
                Yy_FNA = deepcopy(M_FNA[:, end])
                Yy_di = deepcopy(M_di[:, end])
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
                    f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                    z[1,8] = avgScore(f1_5_train, 5)
                    z[1,9] = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    z[1,10] = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    z[1,11] = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                    z[1,12] = f1_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                    z[1,13] = matthews_corrcoef(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                    z[1,14] = recall_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)))
                    z[1,15] = rs
                    z[1,16] = importances["importances_mean"][1]
                    z[1,17] = importances["importances_mean"][2]
                    z[1,18] = importances["importances_mean"][3]
                    z[1,19] = importances["importances_mean"][4]
                    z[1,20] = importances["importances_mean"][5]
                    z[1,21] = importances["importances_mean"][6]
                    z[1,22] = importances["importances_mean"][7]
                    z[1,23] = importances["importances_mean"][8]
                    z[1,24] = importances["importances_mean"][9]
                    z[1,25] = importances["importances_mean"][10]
                    z[1,26] = importances["importances_std"][1]
                    z[1,27] = importances["importances_std"][2]
                    z[1,28] = importances["importances_std"][3]
                    z[1,29] = importances["importances_std"][4]
                    z[1,30] = importances["importances_std"][5]
                    z[1,31] = importances["importances_std"][6]
                    z[1,32] = importances["importances_std"][7]
                    z[1,33] = importances["importances_std"][8]
                    z[1,34] = importances["importances_std"][9]
                    z[1,35] = importances["importances_std"][10]
                else
                    itrain = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    jtrain = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                    ival = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                    jval = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                    println("## CV ##")
                    f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                    traincvtrain = avgScore(f1_5_train, 5) 
                    f1s = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    mccs = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                    rec = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                    f1s2 = f1_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                    mccs2 = matthews_corrcoef(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                    rec2 = recall_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)))
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
                    z = vcat(z, [pnn s c itrain jtrain ival jval traincvtrain f1s mccs rec f1s2 mccs2 rec2 rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10])
                    println(z[end, :])
                end
                println("End of ", itr, " iterations")
                itr += 1
            end
        end
    end
    z_df = DataFrame(Penalty = z[:,1], Solver = z[:,2], C_value = z[:,3], f1_train = z[:,4], mcc_train = z[:,5], f1_ext = z[:,6], mcc_ext = z[:,7], f1_5Ftrain = z[:,8], f1_fna = z[:,9], mcc_fna = z[:,10], recall_fna = z[:,11], f1_di = z[:,12], mcc_di = z[:,13], recall_di = z[:,14], state = z[:,15], im1 = z[:,16], im2 = z[:,17], im3 = z[:,18], im4 = z[:,19], im5 = z[:,20], im6 = z[:,21], im7 = z[:,22], im8 = z[:,23], im9 = z[:,24], im10 = z[:,25], sd1 = z[:,26], sd2 = z[:,27], sd3 = z[:,28], sd4 = z[:,29], sd5 = z[:,30], sd6 = z[:,31], sd7 = z[:,32], sd8 = z[:,33], sd9 = z[:,34], sd10 = z[:,35])
    z_df_sorted = sort(z_df, [:recall_di, :recall_fna, :f1_5Ftrain], rev=true)
    return z_df_sorted
end

## call Logistic Regression ##
optiSearch_df = optimLR(trainDEFSDf, ingestedDEFSDf, extDEFSDf, fnaDEFSDf, diDEFSDf)

## save ##
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\modeling\\hyperparameterTuning_modelSelection_LR1.csv"
CSV.write(savePath, optiSearch_df)


# ==================================================================================================
## define a function for Support Vector Machine ##
function optimSVM(inputDB, inputDB_ingested, inputDB_ext, inputDB_FNA, inputDB_di)

    penalty_r = ["l1", "l2"]  # 2
    loss_r = ["hinge", "squared_hinge"]  # 2
    #gamma_r = ["scale", "auto"] # 2
    #kernel_r = ["linear", "poly", "rbf", "sigmoid"]  # 4
    #c_values_r = vcat(10, 5, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001)  # 15
    #c_values_r = vcat(collect(0.1:0.05:50))  # 15
    c_values_r = vcat(collect(50:0.5:500))  # 15
    
    rs = 42
    z = zeros(1,35)
    itr = 1

    N_train = inputDB
    M_train = inputDB_ingested
    M_ext = inputDB_ext
    M_FNA = inputDB_FNA
    M_di = inputDB_di

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
                    Xx_di = deepcopy(M_di[:, 2:end-1])
                    #
                    Yy_train = deepcopy(M_train[:, end])
                    mm_train = deepcopy(N_train[:, end])
                    Yy_Ext = deepcopy(M_ext[:, end])
                    Yy_FNA = deepcopy(M_FNA[:, end])
                    Yy_di = deepcopy(M_di[:, end])
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
                        f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                        z[1,8] = avgScore(f1_5_train, 5)
                        z[1,9] = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        z[1,10] = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        z[1,11] = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                        z[1,12] = f1_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                        z[1,13] = matthews_corrcoef(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                        z[1,14] = recall_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)))
                        z[1,15] = rs
                        z[1,16] = importances["importances_mean"][1]
                        z[1,17] = importances["importances_mean"][2]
                        z[1,18] = importances["importances_mean"][3]
                        z[1,19] = importances["importances_mean"][4]
                        z[1,20] = importances["importances_mean"][5]
                        z[1,21] = importances["importances_mean"][6]
                        z[1,22] = importances["importances_mean"][7]
                        z[1,23] = importances["importances_mean"][8]
                        z[1,24] = importances["importances_mean"][9]
                        z[1,25] = importances["importances_mean"][10]
                        z[1,26] = importances["importances_std"][1]
                        z[1,27] = importances["importances_std"][2]
                        z[1,28] = importances["importances_std"][3]
                        z[1,29] = importances["importances_std"][4]
                        z[1,30] = importances["importances_std"][5]
                        z[1,31] = importances["importances_std"][6]
                        z[1,32] = importances["importances_std"][7]
                        z[1,33] = importances["importances_std"][8]
                        z[1,34] = importances["importances_std"][9]
                        z[1,35] = importances["importances_std"][10]
                    else
                        itrain = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                        jtrain = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                        ival = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                        jval = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                        println("## CV ##")
                        f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                        traincvtrain = avgScore(f1_5_train, 5) 
                        f1s = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        mccs = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                        rec = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                        f1s2 = f1_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                        mccs2 = matthews_corrcoef(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                        rec2 = recall_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)))
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
                        z = vcat(z, [p l c itrain jtrain ival jval traincvtrain f1s mccs rec f1s2 mccs2 rec2 rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10])
                        println(z[end, :])
                    end
                    println("End of ", itr, " iterations")
                    itr += 1
                end
            end
        end
        z_df = DataFrame(penalty = z[:,1], loss = z[:,2], C_value = z[:,3], f1_train = z[:,4], mcc_train = z[:,5], f1_ext = z[:,6], mcc_ext = z[:,7], f1_5Ftrain = z[:,8], f1_fna = z[:,9], mcc_fna = z[:,10], recall_fna = z[:,11], f1_di = z[:,12], mcc_di = z[:,13], recall_di = z[:,14], state = z[:,15], im1 = z[:,16], im2 = z[:,17], im3 = z[:,18], im4 = z[:,19], im5 = z[:,20], im6 = z[:,21], im7 = z[:,22], im8 = z[:,23], im9 = z[:,24], im10 = z[:,25], sd1 = z[:,26], sd2 = z[:,27], sd3 = z[:,28], sd4 = z[:,29], sd5 = z[:,30], sd6 = z[:,31], sd7 = z[:,32], sd8 = z[:,33], sd9 = z[:,34], sd10 = z[:,35])
        z_df_sorted = sort(z_df, [:recall_di, :recall_fna, :f1_5Ftrain], rev=true)
        return z_df_sorted
    end
end

## call Support Vector Machine ##
optiSearch_df = optimSVM(trainDEFSDf, ingestedDEFSDf, extDEFSDf, fnaDEFSDf, diDEFSDf)

## save ##
savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\XGB_ALL\\modeling\\hyperparameterTuning_modelSelection_SVM1.csv"
CSV.write(savePath, optiSearch_df)


# ==================================================================================================
## define a function for Gradient Boost ##
function optimGradientBoostClass(inputDB, inputDB_ingested, inputDB_ext, inputDB_FNA, inputDB_di)
    #lr_r = vcat(0.3, 0.1)  # 2
    #lr_r = vcat(0.5, collect(1:2:9))  # 6
    #lr_r = vcat(collect(2:0.5:9))  # 15
    lr_r = vcat(collect(3.5:0.1:4.5))  # 11
    #leaf_r = vcat(8, 12, 18)  # 3
    #leaf_r = vcat(collect(2:4:10))  # 3
    leaf_r = vcat(collect(2:1:8))  # 7
    #depth_r = vcat(collect(6:2:10))  # 3
    #depth_r = vcat(collect(4:1:8))  # 5
    depth_r = vcat(collect(5:1:10))  # 6
    #split_r = vcat(collect(10:10:20))  # 2
    #split_r = vcat(collect(15:15:30))  # 2
    #split_r = vcat(30)  # 1
    split_r = vcat(10, 30, 50)  # 3
    #tree_r = vcat(collect(50:100:250))  # 3
    #tree_r = vcat(collect(25:25:75))  # 3
    tree_r = vcat(50)  # 1
    
    rs = 42
    z = zeros(1,37)
    itr = 1

    N_train = inputDB
    M_train = inputDB_ingested
    M_ext = inputDB_ext
    M_FNA = inputDB_FNA
    M_di = inputDB_di

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
                        Xx_di = deepcopy(M_di[:, 2:end-1])
                        #
                        Yy_train = deepcopy(M_train[:, end])
                        mm_train = deepcopy(N_train[:, end])
                        Yy_Ext = deepcopy(M_ext[:, end])
                        Yy_FNA = deepcopy(M_FNA[:, end])
                        Yy_di = deepcopy(M_di[:, end])
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
                            f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                            z[1,10] = avgScore(f1_5_train, 5)
                            z[1,11] = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                            z[1,12] = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                            z[1,13] = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                            z[1,14] = f1_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                            z[1,15] = matthews_corrcoef(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                            z[1,16] = recall_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)))
                            z[1,17] = rs
                            z[1,18] = importances["importances_mean"][1]
                            z[1,19] = importances["importances_mean"][2]
                            z[1,20] = importances["importances_mean"][3]
                            z[1,21] = importances["importances_mean"][4]
                            z[1,22] = importances["importances_mean"][5]
                            z[1,23] = importances["importances_mean"][6]
                            z[1,24] = importances["importances_mean"][7]
                            z[1,25] = importances["importances_mean"][8]
                            z[1,26] = importances["importances_mean"][9]
                            z[1,27] = importances["importances_mean"][10]
                            z[1,28] = importances["importances_std"][1]
                            z[1,29] = importances["importances_std"][2]
                            z[1,30] = importances["importances_std"][3]
                            z[1,31] = importances["importances_std"][4]
                            z[1,32] = importances["importances_std"][5]
                            z[1,33] = importances["importances_std"][6]
                            z[1,34] = importances["importances_std"][7]
                            z[1,35] = importances["importances_std"][8]
                            z[1,36] = importances["importances_std"][9]
                            z[1,37] = importances["importances_std"][10]
                        else
                            itrain = f1_score(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                            jtrain = matthews_corrcoef(Vector(mm_train), predict(reg, Matrix(nn_train)), sample_weight=sampleW)
                            ival = f1_score(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                            jval = matthews_corrcoef(Vector(Yy_Ext), predict(reg, Matrix(Xx_Ext)), sample_weight=sampleExtW)
                            println("## CV ##")
                            f1_5_train = cross_val_score(reg, Matrix(Xx_train), Vector(Yy_train); cv = 5, scoring=f1)
                            traincvtrain = avgScore(f1_5_train, 5) 
                            f1s = f1_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                            mccs = matthews_corrcoef(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)), sample_weight=sampleFNAW)
                            rec = recall_score(Vector(Yy_FNA), predict(reg, Matrix(Xx_FNA)))
                            f1s2 = f1_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                            mccs2 = matthews_corrcoef(Vector(Yy_di), predict(reg, Matrix(Xx_di)), sample_weight=sampleDiW)
                            rec2 = recall_score(Vector(Yy_di), predict(reg, Matrix(Xx_di)))
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
                            z = vcat(z, [lr l t d s itrain jtrain ival jval traincvtrain f1s mccs rec f1s2 mccs2 rec2 rs im1 im2 im3 im4 im5 im6 im7 im8 im9 im10 sd1 sd2 sd3 sd4 sd5 sd6 sd7 sd8 sd9 sd10])
                            println(z[end, :])
                        end
                        println("End of ", itr, " iterations")
                        itr += 1
                    end
                end
            end
        end
    end
    z_df = DataFrame(lr = z[:,1], leaves = z[:,2], trees = z[:,3], depth = z[:,4], minSplit = z[:,5], f1_train = z[:,6], mcc_train = z[:,7], f1_ext = z[:,8], mcc_ext = z[:,9], f1_5Ftrain = z[:,10], f1_fna = z[:,11], mcc_fna = z[:,12], recall_fna = z[:,13], f1_di = z[:,14], mcc_di = z[:,15], recall_di = z[:,16], state = z[:,17], im1 = z[:,18], im2 = z[:,19], im3 = z[:,20], im4 = z[:,21], im5 = z[:,22], im6 = z[:,23], im7 = z[:,24], im8 = z[:,25], im9 = z[:,26], im10 = z[:,27], sd1 = z[:,28], sd2 = z[:,29], sd3 = z[:,30], sd4 = z[:,31], sd5 = z[:,32], sd6 = z[:,33], sd7 = z[:,34], sd8 = z[:,35], sd9 = z[:,36], sd10 = z[:,37])
    z_df_sorted = sort(z_df, [:recall_di, :recall_fna, :f1_5Ftrain], rev=true)
    return z_df_sorted
end

## call Gradient Boost ##
optiSearch_df = optimGradientBoostClass(trainDEFSDf, ingestedDEFSDf, extDEFSDf, fnaDEFSDf, diDEFSDf)

## save ##
savePath = "C:\\Users\\T1208\\PyLAB\\raMSIn\\hyperparameterTuning_modelSelection_GBM4.csv"
CSV.write(savePath, optiSearch_df)
