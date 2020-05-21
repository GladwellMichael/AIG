using Pkg

#This will create a new environment
#Pkg.activate("AIG_Assignment_1")

#This will activate the Assignemnt 1 environemnt 
Pkg.activate(".")

#This will instantiate the environment
Pkg.instantiate()

#The following will update the packages if there is a newer version otherwise they are already installed in the env
Pkg.add("DataFrames")
Pkg.add("CSV")

#This will allow you to use the packages
using CSV
using DataFrames

#This will read the csv file and return a tabular output
CSV.read("bank-additional-full.csv")

#This will read the file and return it as it is from the csv file (without specific columns)
readlines("bank-additional-full.csv")

#This will create a DataFrame called df and return the DataFrame
df = CSV.read("bank-additional-full.csv") |> DataFrame

#Check for inconsistencies with the column names
first(df, 15)

#Check the names of all the columns and check for inconsistencies
names(df)

##-- DATA CLEANING --##
#1. Remove duplicate data
#2. Remove data with unknowns or rows with missing data
#3. Check if each column affects the required output or decision 
#4. Remove unnecessary columns in the dataset that do not help in archieving our goal
#5. change the data types to use only the datatypes usable with the linear regression model

