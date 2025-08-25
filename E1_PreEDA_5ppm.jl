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
df24_top17 = CSV.read("I:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\5ppm\\noSTDtop17.csv", DataFrame)
    top17_4clusters_ = df24_top17[:,2]
    top17_4clusters = ["pixel_id"]
    for top in top17_4clusters_
        push!(top17_4clusters, string(top))
    end
    push!(top17_4clusters, "type")

    df24_train = CSV.read("I:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\5ppm\\df_ROI_for_ML_Opti_train_5ppm.csv", DataFrame)[:,top17_4clusters]
    df24_ext = CSV.read("I:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\5ppm\\df_ROI_for_ML_Opti_ext_5ppm.csv", DataFrame)[:,top17_4clusters]
    df24_ingested = CSV.read("I:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\5ppm\\df_ROI_for_ML_Opti_ingested_5ppm.csv", DataFrame)[:,top17_4clusters]
    df24_FNA = CSV.read("I:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\5ppm\\df_ROI_for_ML_Opti_FNA_5ppm.csv", DataFrame)[:,top17_4clusters]
    df22_DI = CSV.read("I:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\5ppm\\df_ROI_for_ML_Opti_DirectIn_5ppm.csv", DataFrame)[:,top17_4clusters]

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


    for f in 2:size(df24_train, 2)-1
        avg = float(mean(df24_train[:, f]))
        #top = float(maximum(df24_train[:, f]))
        #down = float(minimum(df24_train[:, f]))
        std = float(Statistics.std(df24_train[:, f]))
        for i = 1:size(df24_train, 1)
            #df24_train[i, f] = (df24_train[i, f] - avg) / (top - down)
            df24_train[i, f] = (df24_train[i, f] - avg) / std
        end
    end
    describe(df24_train)
    ## save ##
    savePath = "I:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\5ppm\\df_train_dbMSIn5ppm4nonInDI_STDnorm.csv"
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


    for f in 2:size(df24_ext, 2)-1
        avg = float(mean(df24_ext[:, f]))
        #top = float(maximum(df24_ext[:, f]))
        #down = float(minimum(df24_ext[:, f]))
        std = float(Statistics.std(df24_ext[:, f]))
        for i = 1:size(df24_ext, 1)
            #df24_ext[i, f] = (df24_ext[i, f] - avg) / (top - down)
            df24_ext[i, f] = (df24_ext[i, f] - avg) / std
        end
    end
    describe(df24_ext)
    ## save ##
    savePath = "I:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\5ppm\\df_ext_dbMSIn5ppm4nonInDI_STDnorm.csv"
    CSV.write(savePath, df24_ext)
    
    df24_ext[df24_ext.type .== 1, :]
    # 0: 2943; 1: 3132

# ==================================================================================================
## prepare ingested set ##
size(df24_ingested, 2)
    for row = 1:size(df24_ingested, 1)
        for col = 2:size(df24_ingested, 2)-1
            df24_ingested[row, col] = sqrt(df24_ingested[row, col])
            #df24_ingested[row, col] = log10(df24_ingested[row, col])
        end
    end
    describe(df24_ingested)


    for f in 2:size(df24_ingested, 2)-1
        avg = float(mean(df24_ingested[:, f]))
        #top = float(maximum(df24_ingested[:, f]))
        #down = float(minimum(df24_ingested[:, f]))
        std = float(Statistics.std(df24_ingested[:, f]))
        for i = 1:size(df24_ingested, 1)
            #df24_ingested[i, f] = (df24_ingested[i, f] - avg) / (top - down)
            df24_ingested[i, f] = (df24_ingested[i, f] - avg) / std
        end
    end
    describe(df24_ingested)
    ## save ##
    savePath = "I:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\5ppm\\df_ingested_dbMSIn5ppm4nonInDI_STDnorm.csv"
    CSV.write(savePath, df24_ingested)
    
    df24_ingested[df24_ingested.type .== 1, :]
    # 0: 50392; 1: 46643

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


    for f in 2:size(df24_FNA, 2)-1
        avg = float(mean(df24_FNA[:, f]))
        #top = float(maximum(df24_FNA[:, f]))
        #down = float(minimum(df24_FNA[:, f]))
        std = float(Statistics.std(df24_FNA[:, f]))
        for i = 1:size(df24_FNA, 1)
            #df24_FNA[i, f] = (df24_FNA[i, f] - avg) / (top - down)
            df24_FNA[i, f] = (df24_FNA[i, f] - avg) / std
        end
    end
    describe(df24_FNA)
    ## save ##
    savePath = "I:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\5ppm\\df_FNA_dbMSIn5ppm4nonInDI_STDnorm.csv"
    CSV.write(savePath, df24_FNA)
    
    df24_FNA[df24_FNA.type .== 1, :]
    # 0: 44540; 1: 44161

# ==================================================================================================
# prepare DirectIn set
size(df22_DI, 2)
    for row = 1:size(df22_DI, 1)
        for col = 2:size(df22_DI, 2)-1
            df22_DI[row, col] = sqrt(df22_DI[row, col])
            #df22_DI[row, col] = log10(df22_DI[row, col])
        end
    end
    describe(df22_DI)


    for f in 2:size(df22_DI, 2)-1
        avg = float(mean(df22_DI[:, f]))
        #top = float(maximum(df22_DI[:, f]))
        #down = float(minimum(df22_DI[:, f]))
        std = float(Statistics.std(df22_DI[:, f]))
        for i = 1:size(df22_DI, 1)
            #df22_DI[i, f] = (df22_DI[i, f] - avg) / (top - down)
            df22_DI[i, f] = (df22_DI[i, f] - avg) / std
        end
    end
    describe(df22_DI)
    ## save ##
    savePath = "I:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_Ingested4ALL\\5ppm\\df_nonInDI_dbMSIn5ppm4nonInDI_STDnorm.csv"
    CSV.write(savePath, df22_DI)
    
    df22_DI[df22_DI.type .== 0, :]
    # 0: 1513; 1: 1515
