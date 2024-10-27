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
df24_top18 = CSV.read("H:\\3_output_raMSIn\\noSTDtop18_NonInFNA.csv", DataFrame)
    top18_4clusters_ = df24_top18[:,2]
    top18_4clusters = ["pixel_id"]
    for top in top18_4clusters_
        push!(top18_4clusters, string(top))
    end
    push!(top18_4clusters, "type")

    pop_ = df24_top18[:,3]
    pop = ["pixel_id"]
    for p in pop_
        push!(pop, string(p))
    end
    push!(pop, "type")

    df24_FNA = CSV.read("H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_NonIn\\df_ROI4_for_ML_Opti_NonInFNA_all.csv", DataFrame)
    df_FNA = DataFrame(df24_FNA[:, top18_4clusters[vcat(1,end)]])
    
    count = 2
    for top in top18_4clusters[2:19]
        if (top in names(df24_FNA)) == true
            insertcols!(df_FNA, count, (top => df24_FNA[:, top]))
            count += 1
        elseif (pop[count] in names(df24_FNA)) == true && (pop[count] !== "100.9462")
            insertcols!(df_FNA, count, (top => df24_FNA[:, pop[count]]))
            count += 1
        else
            insertcols!(df_FNA, count, (top .=> float(0)))
            count += 1
        end
    end
    describe(df_FNA)

    savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_NonIn\\df_NonInFNA24_all.csv"
    CSV.write(savePath, df_FNA)


# ==================================================================================================
# prepare FNA set
size(df_FNA, 2)
    for row = 1:size(df_FNA, 1)
        for col = 2:size(df_FNA, 2)-1
            df_FNA[row, col] = sqrt(df_FNA[row, col])
            #df_FNA[row, col] = log10(df_FNA[row, col])
        end
    end
    describe(df_FNA)

    ## save ##
    savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_NonIn\\df_NonInFNA24_alllog.csv"
    CSV.write(savePath, df_FNA)

    for f in 2:size(df_FNA, 2)-1
        avg = float(mean(df_FNA[:, f]))
        top = float(maximum(df_FNA[:, f]))
        down = float(minimum(df_FNA[:, f]))
        for i = 1:size(df_FNA, 1)
            if (top - down) == 0
                df_FNA[i, f] = float(0)
            else
                df_FNA[i, f] = (df_FNA[i, f] - avg) / (top - down)
            end
        end
    end
    describe(df_FNA)
    ## save ##
    savePath = "H:\\3_output_raMSIn\\3_3_Output_raMSIn_HKU_NonIn\\df_NonInFNA24_allstd.csv"
    CSV.write(savePath, df_FNA)
    
    df_FNA[df_FNA.type .== 1, :]
    # 0: 44540; 1: 10302
