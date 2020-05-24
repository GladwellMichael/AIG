using CSV
using Statistics
using  PyCall
using PrettyTables


#reading dataset for experimenting purpose
df =CSV.read("ex2data1.csv")
# dat =CSV.read("bank-additional-full.csv")

#converting dataframe to matrix
df = convert(Matrix,df)
# dat = convert(Matrix,dat)

# function for standadising the data
function standadiation()   
end

#function for cleaning data
############################################
function datacleaning(data)

#getting the length of the data set
x = size(data)[1]

#looping over the  dataset
for i in 1:x
    
# cotegorising the job title column
if(data[i,2]=="housemaid")
    data[i,2]=1
elseif(data[i,2]=="services")
    data[i,2]=2
elseif(data[i,2]=="admin.")
    data[i,2]=3
elseif(data[i,2]=="blue-collar")
    data[i,2]=4
elseif(data[i,2]=="technician")
    data[i,2]= 5
elseif(data[i,2]=="retired")
    data[i,2]=6
elseif(data[i,2]=="management")
    data[i,2]=7
elseif(data[i,2]=="unemployed")
    data[i,2]=8
elseif(data[i,2]=="entrepreneur")
    data[i,2]=9
elseif(data[i,2]=="self-employed")
    data[i,2]=10
elseif(data[i,2]=="student")
    data[i,2]=11
elseif(data[i,2]=="unknown")
    data[i,2]=0
                                        
end
#catergorising marrital status
if(data[i,3]=="unknown")
    data[i,3]=0
elseif(data[i,3]=="single")
    data[i,3]=1
elseif(data[i,3]=="married")
    data[i,3]=2
elseif(data[i,3]=="divorced")
    data[i,3]=4
end


#categorising  education  status
if(data[i,4]=="unknown")
    data[i,4]=0
elseif(data[i,4]=="high.school")
    data[i,4]=1
elseif(data[i,4]=="basic.6y")
    data[i,4]=2
elseif(data[i,4]=="basic.9y")
    data[i,4]=3
elseif(data[i,4]=="professional.course")
    data[i,4]=4
elseif(data[i,4]=="basic.4y")
    data[i,4]=5
elseif(data[i,4]=="university.degree")
    data[i,4]=6
end

#categorising default  column
if(data[i,5]=="no")
    data[i,5]=1
elseif(data[i,5]=="unknown")
    data[i,5]=2
end

#categorising housing column
if(data[i,6]=="yes")
    data[i,6]=1
elseif(data[i,6]=="no")
    data[i,6]=2
elseif(data[i,6]=="unknown")
    data[i,6]= 0
end

#categorising loan column
if(data[i,7]=="yes")
    data[i,7]=1
elseif(data[i,7]=="no")
    data[i,7]=2
elseif(data[i,7]=="unknown")
    data[i,7]=0
end

#categorising contact
if(data[i,8]=="telephone")
    data[i,8]=1
elseif(data[i,8]=="cellular")
    data[i,8]=2
end

#catergorising month
if(data[i,9]=="may")
    data[i,9]=1
elseif(data[i,9]=="jun")
    data[i,9]=2
elseif(data[i,9]=="jul")
    data[i,9]=3
elseif(data[i,9]=="aug")
    data[i,9]=4
elseif(data[i,9]=="oct")
    data[i,9]=5
elseif(data[i,9]=="nov")
    data[i,9]=6
elseif(data[i,9]=="dec")
    data[i,9]=7
elseif(data[i,9]=="mar")
    data[i,9]=8
elseif(data[i,9]=="apr")
    data[i,9]=9
elseif(data[i,9]=="sep")
    data[i,9]=10
end

#categorising day of the week
if(data[i,10]=="mon")
    data[i,10]=1
elseif(data[i,10]=="tue")
    data[i,10]=2
elseif(data[i,10]=="wed")
    data[i,10]=3
elseif(data[i,10]=="thu")
    data[i,10]=4
elseif(data[i,10]=="fri")
    data[i,10]=5
end

#categorising poutcome column
if(data[i,15]=="nonexistent")
    data[i,15]=1
elseif(data[i,15]=="failure")
    data[i,15]=2
elseif(data[i,15]=="success")
    data[i,15]=3
end

# categorising the label (0 and 1)
if(data[i,21]=="yes")
    data[i,21]=1
elseif(data[i,21]=="no")
    data[i,21]=0
end

end
    return data
end
############################################
# clean = datacleaning(dat)
# print(clean)
# pretty_table(clean)



# data for experimenting
#splting data into  features(X) and labels(y)
X = df[:,1:2]
y =df[:,end]

#spliting data into training(80%) and testing(20%)
# X traing and y traing
X_train = X[1:60,:]
y_train = y[1:60,:]

#x testing and y testing
X_test = X[61:end,:]
y_test = y[61:end,:]


# sigmoid function
function sigmoid(z)
    return 1 /(1+ exp(-z[1]))
end

# hypothsesis function for calculating predicting y
function hypothesis(z)
    return sigmoid(z)
end


# cost function for computing the cost of all taining sample
function regularised_cost_function(features,y,theta,lambda)
    m ,n= size(features)
    estimate = hypothesis(features*theta)
    # print("estimate",estimate)
    # print("Length",length(y))
    # print(size(features))
    # print(theta)
    total_cost =((-y)' * log.(estimate)).-((1 .- y)' * log.(1 .- estimate))
                .+(lambda/(2*m) * sum(theta[2 : end] .^ 2))
    
    # derivative = (1/m) * (X') * (estimate.-y) + ((1/m) * (lambda * theta))
    # print("Derivative",derivative)
    derivative =  (1/m) * (features[:, 1])' * (estimate.-y)
    # print("Derivative",derivative)
    return  ( total_cost,derivative)
end


function fit(X,y,lambda,learning_rate,number_of_iterations= 800)
    length=size(X)[2]
    # print(length)
    theta = zeros(length)  
    # print(theta)
    # print(size(X))
    # print(size(y))
    cost= zeros(number_of_iterations,60)
    for i in 1:number_of_iterations
        cost[i,:],gradient = regularised_cost_function(X,y,theta,lambda)
        theta = theta .- (learning_rate * gradient)
        
    end
    return (theta,cost)
end

#function for making prediction on unseen adata
############################################
function predict(x)
    theta,cost  = fit(X_train,y_train,0.001,10)
    probability = zeros(size(x)[1])
    for i in 1:(size(x)[1])
        probability[i]= hypothesis((X[i,:])'*theta)
    end
    
    return probability
end
############################################

prediction = predict(X_test)

#confusion matrix (accurace,pecision,recall)
############################################
function accurace_testing(estimate , actual)
    counter = 0
    accurace = 0
    for i in 1:(length(actual))
        if(estimate[i]==actual[i])
            counter = counter+1
        end
    end
    accurace = (counter / length(actual))*100
    return accurace
end
############################################
accurace = accurace_testing(prediction,y_test)

print("Pred ",prediction)
print("accurace ",accurace)