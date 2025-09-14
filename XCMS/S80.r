if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("xcms")

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("MSnbase")

library(xcms)

f.in <- list.files(pattern = '.(mz[X]{0,1}ML|cdf)', recursive = TRUE)

xset <- xcms::xcmsSet(f.in, method = "centWave", ppm = 15, 
                      snthr = 10, peakwidth = c(5, 40), mzdiff = 0.01, 
                      nSlaves = 12)

##retention time correction 
pdf('rector-obiwarp.pdf') 

xsetc <-  xcms::retcor(xset, method = "obiwarp", plottype = "deviation", 
                       profStep = 0.1) 
dev.off() 

## peak grouping 
xset2 <- xcms::group(xsetc, bw = 5, mzwid = 0.015, minfrac = 0.5) 

##gap filling 
xset3 <- xcms::fillPeaks(xset2) 

##peak table outputting 
values <- xcms::groupval(xset3, "medret", value = "into") 
values.maxo <- xcms::groupval(xset3, "medret", value = 'maxo') 
values.maxint <- apply(values.maxo, 1, max) 
peak.table <- cbind(name = xcms::groupnames(xset3), 
                    groupmat = xcms::groups(xset3), 
                    maxint = values.maxint, 
                    values) 
rownames(peak.table) <- NULL 
write.csv(peak.table, "Peak-table.csv", row.names = FALSE)

