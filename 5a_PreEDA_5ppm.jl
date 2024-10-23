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
df24_top18 = CSV.read("H:\\3_output_raMSIn\\noSTDtop18_5ppm.csv", DataFrame)
    top18_4clusters_ = df24_top18[:,2]
    top18_4clusters = ["pixel_id"]
    for top in top18_4clusters_
        push!(top18_4clusters, string(top))
    end
    push!(top18_4clusters, "type")

    df24_train = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_ROI4_for_ML_Opti_train.csv", DataFrame)[:,top18_4clusters]
    df24_ext = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_ROI4_for_ML_Opti_ext.csv", DataFrame)[:,top18_4clusters]
    df24_FNA = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_ROI4_for_ML_Opti_FNA.csv", DataFrame)[:,top18_4clusters]

    df_pixels_train = CSV.read("H:\\3_output_raMSIn\\df_train24.csv", DataFrame)
    pixels_train = Array(df_pixels_train[:, "pixel_id"])
    df_pixels_ext = CSV.read("H:\\3_output_raMSIn\\df_ext24.csv", DataFrame)
    pixels_ext = Array(df_pixels_ext[:, "pixel_id"])
    
    df24_train = df24_train[in.(df24_train.pixel_id, (pixels_train,)), :]
    df24_ext = df24_ext[in.(df24_ext.pixel_id, (pixels_ext,)), :]
    df24_FNA = df24_FNA[in.(df24_FNA.pixel_id, (pixels_FNA,)), :]

    savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_train24.csv"
    CSV.write(savePath, df24_train)
    savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_ext24.csv"
    CSV.write(savePath, df24_ext)
    savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_FNA24.csv"
    CSV.write(savePath, df24_FNA)


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
    savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_train24_log.csv"
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
    savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_train24_std.csv"
    CSV.write(savePath, df24_train)
    
    df24_train[df24_train.type .== 1, :]
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
    savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_ext24_log.csv"
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
    savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_ext24_std.csv"
    CSV.write(savePath, df24_ext)
    
    df24_ext[df24_ext.type .== 1, :]
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
    savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_FNA24_log.csv"
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
    savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSI_HKU_Ingested4FNA\\0_dbMSI_5ppm\\df5ppm_FNA24_std.csv"
    CSV.write(savePath, df24_FNA)
    
    df24_FNA[df24_FNA.type .== 1, :]
    # 0: 44540; 1: 44161
