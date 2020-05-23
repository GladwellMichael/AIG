using Pkg

#This will create a new environment
#Pkg.activate("AIG_Assignment_1")

#This will activate the Assignemnt 1 environemnt 
Pkg.activate(".")

#This will instantiate the environment
Pkg.instantiate()

#The following will update the packages if there is a newer version otherwise they are already installed in the env
Pkg.add("CSV")
Pkg.add("Plots")
Pkg.add("PyPlot")
Pkg.add("PyCall")
Pkg.add("Seaborn")
Pkg.add("DataFrames")
Pkg.add("Statistics")


#This will allow you to use the packages
using CSV
using Plots
using PyCall
using PyPlot
using PyCall
using Seaborn
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
#Remove duplicate rows from the DataFramedf 
df = unique(df);

#2. Remove data with unknowns or rows with missing data
#https://towardsdatascience.com/machine-learning-case-study-a-data-driven-approach-to-predict-the-success-of-bank-telemarketing-20e37d46c31c
#python code to check the ratio of y and n for the subject of the project 
  plt.figure(figsize=(8,6))
  Y = data["y"]
  total = len(Y)*1.
  ax=sns.countplot(x="y", data=data)
  for p in ax.patches:
    ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))

  #put 11 ticks (therefore 10 steps), from 0 to the total number of rows in the dataframe
  ax.yaxis.set_ticks(np.linspace(0, total, 11))
  #adjust the ticklabel to the desired format, without changing the position of the ticks.
  ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))
  ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
  # ax.legend(labels=["no","yes"])
  plt.show()

#python code check if a column is necessary
    import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

def countplot_withY(label, dataset):
  plt.figure(figsize=(20,10))
  Y = data[label]
  total = len(Y)*1.
  ax=sns.countplot(x=label, data=dataset, hue="y")
  for p in ax.patches:
    ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))

  #put 11 ticks (therefore 10 steps), from 0 to the total number of rows in the dataframe
  ax.yaxis.set_ticks(np.linspace(0, total, 11))
  #adjust the ticklabel to the desired format, without changing the position of the ticks.
  ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))
  ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
  # ax.legend(labels=["no","yes"])
  plt.show()
    
#3. Check if each column affects the required output or decision 
#4. Remove unnecessary columns in the dataset that do not help in archieving our goal
#5. change the data types to use only the datatypes usable with the linear regression model

