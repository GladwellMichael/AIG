#Create a New Environment for the Project 
#mkdir("AIG Assignment 1") -> This will create the folder in the default Julia env folder
#cd("AIG Assignment 1")
#]
#activate .
#st

#import Pkg

#Pkg.add("DataFrames")
#Pkg.add("CSV")

using Dataframes
using CSV

#To read the file as CSV, we need to clean the data contained in the file. We can read the file using readline
readlines("bank-additional-full.csv")

#Create a DataFrame from the csv file
dataSet = CSV.read("bank-additional-full.csv") |> DataFrame

