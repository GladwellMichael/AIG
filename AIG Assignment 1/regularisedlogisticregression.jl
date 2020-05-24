using CSV
using Statistics
using  PyCall
using PrettyTables


#reading dataset for experimenting purpose
# df =CSV.read("ex2data1.csv")
da =CSV.read("bank-additional-full.csv")


function  datacleaning(da)
#removing duplicates
dat = unique(da)

#removing some of the column
delete!(dat,:age)
delete!(dat,:contact)
delete!(dat,:day_of_week)
delete!(dat,:month)
delete!(dat,:marital)
return dat
end

dat = datacleaning(da)

#converting dataframe to matrix
# df = convert(Matrix,df)
dat = convert(Matrix,dat)



# function for standadising the x train
function features_scaling(x_train)   
    meal = mean(x_train, dims =  1)
    standard_deviation = std(x_train,dims = 1)
    standardised_x = (x_train .- mean)/ standard_deviation
    return (standardised_x , meal , standard_deviation)
end


#function for normalising the x test 
function transform_features(x_test ,mean , standard_deviation)
    transforned_x = (x_test .- mean) ./ standard_deviation
    return transforned_x

end





#function for cleaning data
############################################
function categorisingData(data)

#getting the length of the data set
x = size(data)[1]

#looping over the  dataset
for i in 1:x
    
# cotegorising the job title column
if(data[i,1]=="housemaid")
    data[i,1]=1
elseif(data[i,1]=="services")
    data[i,1]=2
elseif(data[i,1]=="admin.")
    data[i,1]=3
elseif(data[i,1]=="blue-collar")
    data[i,1]=4
elseif(data[i,1]=="technician")
    data[i,1]= 5
elseif(data[i,1]=="retired")
    data[i,1]=6
elseif(data[i,1]=="management")
    data[i,1]=7
elseif(data[i,1]=="unemployed")
    data[i,1]=8
elseif(data[i,1]=="entrepreneur")
    data[i,1]=9
elseif(data[i,1]=="self-employed")
    data[i,1]=10
elseif(data[i,1]=="student")
    data[i,1]=11
elseif(data[i,1]=="unknown")
    data[i,1]=0
                                        
end



#categorising  education  status
if(data[i,2]=="unknown")
    data[i,2]=0
elseif(data[i,2]=="high.school")
    data[i,2]=1
elseif(data[i,2]=="basic.6y")
    data[i,2]=2
elseif(data[i,2]=="basic.9y")
    data[i,2]=3
elseif(data[i,2]=="professional.course")
    data[i,2]=4
elseif(data[i,2]=="basic.4y")
    data[i,2]=5
elseif(data[i,2]=="university.degree")
    data[i,2]=6
end

#categorising default  column
if(data[i,3]=="no")
    data[i,3]=1
elseif(data[i,3]=="unknown")
    data[i,3]=2
end

#categorising housing column
if(data[i,4]=="yes")
    data[i,4]=1
elseif(data[i,4]=="no")
    data[i,4]=2
elseif(data[i,4]=="unknown")
    data[i,4]= 0
end

#categorising loan column
if(data[i,5]=="yes")
    data[i,5]=1
elseif(data[i,5]=="no")
    data[i,5]=2
elseif(data[i,5]=="unknown")
    data[i,5]=0
end


#categorising poutcome column
if(data[i,10]=="nonexistent")
    data[i,10]=1
elseif(data[i,10]=="failure")
    data[i,10]=2
elseif(data[i,10]=="success")
    data[i,10]=3
end

# categorising the label (0 and 1)
if(data[i,16]=="yes")
    data[i,16]=1
elseif(data[i,16]=="no")
    data[i,16]=0
end

end
    return data
end
############################################

#calling the function for categorising data
df = categorisingData(dat)


# data for experimenting
#splting data into  features(X) and labels(y)
X = df[:,1:16]
y =df[:,end]

#spliting data into training(80%) and testing(20%)
# X traing and y traing
X_train = X[1:32950,:]
y_train = y[1:32950,:]

#x testing and y testing
X_test = X[32951:end,:]
y_test = y[32951:end,:]

#scaling the  X train #
# X_train, mean ,standard_deviation = features_scaling(X_train)

# #normalising the x tesing
# X_test = transform_features(X_test, mean , standard_deviation)



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

    total_cost = ((-y)' * log.(estimate))-((1 .- y)' * log.(1 .- estimate)) .+  (lambda/(2*m) *  broadcast(abs, theta))
    
    # derivative = (1/m) * (X') * (estimate.-y) + ((1/m) * (lambda * theta))
    # print("Derivative",derivative)
    derivative =  (1/m) * (features[:, 1])' * (estimate.-y).+  (lambda/(2*m) *  broadcast(abs, theta))
    # print("Derivative ",derivative)
    return  ( total_cost, derivative)
end


function fit(X,y,lambda,learning_rate,number_of_iterations= 800)
    length=size(X)[2]
    
    # print(length)
    theta = zeros(length)  
   
    cost= zeros(number_of_iterations,2950)
    for i in 1:number_of_iterations
        cost[i,:],gradient = regularised_cost_function(X[i,:],y,theta,lambda)
        theta = theta .- (learning_rate * gradient)
        
    end
    # print("Theta :",cost)
    return (theta,cost)
end

theta,cost  = fit(X_train,y_train,0.001,10)


# #function for making prediction on unseen adata
# ############################################
function predict(xx)
    theta,cost  = fit(X_train,y_train,0.001,10)
    print(size(theta))
    print(size(xx))
    probability = zeros(size(xx)[1])
    
    for i in 1:(size(xx)[1])
        result = transpose(xx[i, :]) .* theta
        print("Result :",result)
        probability[i]= sigmoid(result)
    end
    
    return probability
end
############################################

prediction = predict(X_test)
print("Predictions :",prediction)


# #confusion matrix (accurace,pecision,recall)
# ############################################
function confusion_matrix(estimate , actual)

    positive_class = Int64[]
    negative_class = Int64[]
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    counter = 0
    recall = 0
    precision = 0 
    accurace = 0


    for i in 1:(length(actual))
        #splitting positive and negative elments
        if(actual[i]==1.0)
            push!(positive_class,actual[i])
        else
            push!(negative_class,actual[i])
        end

       
        if(estimate[i]==1.0 )
            if(actual[i]== estimate[i])
            print("\n",i)
            true_positive = true_positive +1 
            elseif(actual[i] != estimate[i])
            false_negative = false_negative +1
            end
        end

        #getting TN and FP
        if(estimate[i]== 0 )
            if(actual[i]== estimate[i])
            print("\n",i)
            true_negative = true_negative +1 
            elseif(actual[i] != estimate[i])
            false_positive = false_positive +1
            end
        end

    end
    # TP +  TN /  total number of test samples
    accurace = ((true_positive + true_negative)/length(actual))*100
    #TP / TP + FN
    recall = ((true_positive) /(true_positive + false_negative))*100
    #TP / TP + FP
    precision = ((true_positive) /(  true_positive + false_positive))*100
    #returning a tuple
    return (accurace ,recall ,precision)
end
############################################

accurace ,recall ,precision = confusion_matrix(prediction,y_test)

print("\nrecall :",recall)
print("\naccurace :",accurace)
print("\nprecision  :",precision)