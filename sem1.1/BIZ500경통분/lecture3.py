# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% vscode={"languageId": "r"}
rm(list=ls())

setwd("./")

# %% vscode={"languageId": "r"}
longdist <- read.csv('./ch1andch2_dataset/long_distance_telephone_bills.csv')
head( longdist )

# %% vscode={"languageId": "r"}
bills <- longdist$Bills

# %% vscode={"languageId": "r"}
mean( bills )
median( bills )
range( bills )
diff( range( bills ) )
var( bills )
sd( bills )
sd( bills )/mean( bills )

# %% vscode={"languageId": "r"}
quantile( bills )
quantile( bills , prob=c(0.3, 0.8))

# %% vscode={"languageId": "r"}
bills <- longdist$Bills
times <- read.csv('./ch3_dataset/serving_times_of_drive_throughs.csv')
head( times )
boxplot( times, horizontal=TRUE, col="lightblue" )

# %% vscode={"languageId": "r"}
scores <- read.csv('./ch3_dataset/GMAT_and_GPA_scores_for_MBA_students.csv')
head( scores )
gmat <- scores$GMAT
gpa <- scores$GPA
plot( gmat, gpa )

# %% vscode={"languageId": "r"}
cov( gmat, gpa )
cor( gmat, gpa )

# %% vscode={"languageId": "r"}
