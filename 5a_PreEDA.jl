VERSION
## install packages needed ##
using Pkg
#Pkg.add("ScikitLearn")
#Pkg.add(PackageSpec(url=""))

## import packages from Julia ##
using CSV, DataFrames, Conda, LinearAlgebra, Statistics
using ProgressBars
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split

## import all data ##
df24_top18 = CSV.read("G:\\raMSIn\\XGB_Importance2\\noSTDtop18.csv", DataFrame)
    top18_4clusters_ = df24_top18[:,2]
    top18_4clusters = ["pixel_id"]
    for top in top18_4clusters_
        push!(top18_4clusters, string(top))
    end
    push!(top18_4clusters, "type")

    df24_train = CSV.read("G:\\raMSIn\\XGB_FNA\\df_ROI4_for_ML_Opti_train.csv", DataFrame)[:,top18_4clusters]
    df24_ext = CSV.read("G:\\raMSIn\\XGB_FNA\\df_ROI4_for_ML_Opti_ext.csv", DataFrame)[:,top18_4clusters]
    df24_FNA = CSV.read("G:\\raMSIn\\XGB_FNA\\df_ROI4_for_ML_Opti_FNA.csv", DataFrame)[:,top18_4clusters]

    savePath = "G:\\raMSIn\\XGB_Importance2\\df_train24.csv"
    CSV.write(savePath, df24_train)
    savePath = "G:\\raMSIn\\XGB_Importance2\\df_ext24.csv"
    CSV.write(savePath, df24_ext)
    savePath = "G:\\raMSIn\\XGB_Importance2\\df_FNA24.csv"
    CSV.write(savePath, df24_FNA)


df22_top18 = CSV.read("G:\\raMSIn\\XGB_Importance\\noSTDtop18.csv", DataFrame)
    top18_2clusters_ = df22_top18[:,2]
    top18_2clusters = ["pixel_id"]
    for top in top18_2clusters_
        push!(top18_2clusters, string(top))
    end
    push!(top18_2clusters, "type")
    
    df22_train = CSV.read("G:\\raMSIn\\XGB_FNA\\df_ROI_for_ML_Opti_train.csv", DataFrame)[:,top18_2clusters]
    df22_ext = CSV.read("G:\\raMSIn\\XGB_FNA\\df_ROI_for_ML_Opti_ext.csv", DataFrame)[:,top18_2clusters]
    df22_FNA = CSV.read("G:\\raMSIn\\XGB_FNA\\df_ROI_for_ML_Opti_FNA.csv", DataFrame)[:,top18_2clusters]

    savePath = "G:\\raMSIn\\XGB_Importance\\df_train22.csv"
    CSV.write(savePath, df22_train)
    savePath = "G:\\raMSIn\\XGB_Importance\\df_ext22.csv"
    CSV.write(savePath, df22_ext)
    savePath = "G:\\raMSIn\\XGB_Importance\\df_FNA22.csv"
    CSV.write(savePath, df22_FNA)

# ==================================================================================================
## prepare training set ##
size(df24_train, 2)
    for row = 1:size(df24_train, 1)
        for col = 2:size(df24_train, 2)-1
            df24_train[row, col] = sqrt(df24_train[row, col])
            #df24_train[row, col] = log10(df24_train[row, col])
        end
    end
    describe(df24_train)

    ## save ##
    savePath = "G:\\raMSIn\\XGB_Importance2\\df_train24_log.csv"
    CSV.write(savePath, df24_train)

    for f in 2:size(df24_train, 2)-1
        avg = float(mean(df24_train[:, f]))
        top = float(maximum(df24_train[:, f]))
        down = float(minimum(df24_train[:, f]))
        for i = 1:size(df24_train, 1)
            df24_train[i, f] = (df24_train[i, f] - avg) / (top - down)
        end
    end
    describe(df24_train)
    ## save ##
    savePath = "G:\\raMSIn\\XGB_Importance2\\df_train24_std.csv"
    CSV.write(savePath, df24_train)
    
    df24_train[df24_train.type .== 1, :]
    # 0: 47449; 1: 43511
##
size(df22_train, 2)
    for row = 1:size(df22_train, 1)
        for col = 2:size(df22_train, 2)-1
            df22_train[row, col] = sqrt(df22_train[row, col])
            #df22_train[row, col] = log10(df22_train[row, col])
        end
    end
    describe(df22_train)

    ## save ##
    savePath = "G:\\raMSIn\\XGB_Importance\\df_train22_log.csv"
    CSV.write(savePath, df22_train)

    for f in 2:size(df22_train, 2)-1
        avg = float(mean(df22_train[:, f]))
        top = float(maximum(df22_train[:, f]))
        down = float(minimum(df22_train[:, f]))
        for i = 1:size(df22_train, 1)
            df22_train[i, f] = (df22_train[i, f] - avg) / (top - down)
        end
    end
    describe(df22_train)
    ## save ##
    savePath = "G:\\raMSIn\\XGB_Importance\\df_train22_std.csv"
    CSV.write(savePath, df22_train)
    
    df22_train[df22_train.type .== 1, :]
    # 0: 47449; 1: 43511


# ==================================================================================================
## prepare ext val set ##
size(df24_ext, 2)
    for row = 1:size(df24_ext, 1)
        for col = 2:size(df24_ext, 2)-1
            df24_ext[row, col] = sqrt(df24_ext[row, col])
            #df24_ext[row, col] = log10(df24_ext[row, col])
        end
    end
    describe(df24_ext)

    ## save ##
    savePath = "G:\\raMSIn\\XGB_Importance2\\df_ext24_log.csv"
    CSV.write(savePath, df24_ext)

    for f in 2:size(df24_ext, 2)-1
        avg = float(mean(df24_ext[:, f]))
        top = float(maximum(df24_ext[:, f]))
        down = float(minimum(df24_ext[:, f]))
        for i = 1:size(df24_ext, 1)
            df24_ext[i, f] = (df24_ext[i, f] - avg) / (top - down)
        end
    end
    describe(df24_ext)
    ## save ##
    savePath = "G:\\raMSIn\\XGB_Importance2\\df_ext24_std.csv"
    CSV.write(savePath, df24_ext)
    
    df24_ext[df24_ext.type .== 1, :]
    # 0: 2943; 1: 3132
##
size(df22_ext, 2)
    for row = 1:size(df22_ext, 1)
        for col = 2:size(df22_ext, 2)-1
            df22_ext[row, col] = sqrt(df22_ext[row, col])
            #df22_ext[row, col] = log10(df22_ext[row, col])
        end
    end
    describe(df22_ext)

    ## save ##
    savePath = "G:\\raMSIn\\XGB_Importance\\df_ext22_log.csv"
    CSV.write(savePath, df22_ext)

    for f in 2:size(df22_ext, 2)-1
        avg = float(mean(df22_ext[:, f]))
        top = float(maximum(df22_ext[:, f]))
        down = float(minimum(df22_ext[:, f]))
        for i = 1:size(df22_ext, 1)
            df22_ext[i, f] = (df22_ext[i, f] - avg) / (top - down)
        end
    end
    describe(df22_ext)
    ## save ##
    savePath = "G:\\raMSIn\\XGB_Importance\\df_ext22_std.csv"
    CSV.write(savePath, df22_ext)
    
    df22_ext[df22_ext.type .== 1, :]
    # 0: 2943; 1: 3132


# ==================================================================================================
# prepare FNA set
size(df24_FNA, 2)
    for row = 1:size(df24_FNA, 1)
        for col = 2:size(df24_FNA, 2)-1
            df24_FNA[row, col] = sqrt(df24_FNA[row, col])
            #df24_FNA[row, col] = log10(df24_FNA[row, col])
        end
    end
    describe(df24_FNA)

    ## save ##
    savePath = "G:\\raMSIn\\XGB_Importance2\\df_FNA24_log.csv"
    CSV.write(savePath, df24_FNA)

    for f in 2:size(df24_FNA, 2)-1
        avg = float(mean(df24_FNA[:, f]))
        top = float(maximum(df24_FNA[:, f]))
        down = float(minimum(df24_FNA[:, f]))
        for i = 1:size(df24_FNA, 1)
            df24_FNA[i, f] = (df24_FNA[i, f] - avg) / (top - down)
        end
    end
    describe(df24_FNA)
    ## save ##
    savePath = "G:\\raMSIn\\XGB_Importance2\\df_FNA24_std.csv"
    CSV.write(savePath, df24_FNA)
    
    df24_FNA[df24_FNA.type .== 1, :]
    # 0: 44540; 1: 44161
##
size(df22_FNA, 2)
    for row = 1:size(df22_FNA, 1)
        for col = 2:size(df22_FNA, 2)-1
            df22_FNA[row, col] = sqrt(df22_FNA[row, col])
            #df22_FNA[row, col] = log10(df22_FNA[row, col])
        end
    end
    describe(df22_FNA)

    ## save ##
    savePath = "G:\\raMSIn\\XGB_Importance\\df_FNA22_log.csv"
    CSV.write(savePath, df22_FNA)

    for f in 2:size(df22_FNA, 2)-1
        avg = float(mean(df22_FNA[:, f]))
        top = float(maximum(df22_FNA[:, f]))
        down = float(minimum(df22_FNA[:, f]))
        for i = 1:size(df22_FNA, 1)
            df22_FNA[i, f] = (df22_FNA[i, f] - avg) / (top - down)
        end
    end
    describe(df22_FNA)
    ## save ##
    savePath = "G:\\raMSIn\\XGB_Importance\\df_FNA22_std.csv"
    CSV.write(savePath, df22_FNA)
    
    df22_FNA[df22_FNA.type .== 1, :]
    # 0: 44540; 1: 44161
