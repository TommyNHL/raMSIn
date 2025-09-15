if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("xcms")

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("MSnbase")

library(xcms)

f.in <- list.files(pattern = '.(mz[X]{0,1}ML|cdf)', recursive = TRUE)

xset <- xcms::xcmsSet(f.in, method = "centWave", ppm = 25, 
                      snthr = 10, peakwidth = c(5, 50), mzdiff = 0.01, 
                      nSlaves = 12)

##retention time correction 
pdf('rector-obiwarp.pdf') 

xsetc <-  xcms::retcor(xset, method = "obiwarp", plottype = "deviation", 
                       profStep = 0.1) 
dev.off() 

peak <- xset@peaks
write.csv(peak, "Peak-table.csv", row.names = FALSE)

