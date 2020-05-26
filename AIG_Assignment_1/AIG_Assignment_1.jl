using CSV ,DataFrames
using Statistics


#reading dataset for experimenting purpose
# df =CSV.read("ex2data1.csv")
da =CSV.read("bank-additional-full.csv")

#fanction for cleaning dataset
function  datacleaning(da)

#removing duplicates
dat = unique(da)

#removing some of the column
    select!(dat,Not(:age))
    select!(dat,Not(:contact))
    select!(dat,Not(:day_of_week))
    select!(dat,Not(:month))
    select!(dat,Not(:marital))
return dat
end

#envoking  datacleaning method
dat = datacleaning(da)

#converting dataframe to matrix
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


#function for categorising data
function categorisingData(data)

     for i = 1:(size(data)[2])
        if(i == size(data)[2])
            for j = 1:size(data)[1]
                if(data[j, i] == "yes")
                    data[j, i] = Int(1)
                else
                    data[j, i] = Int(0)
                end
            end
        else
            identical = unique(data[:,i])
            for j = 1:size(data)[1]
                for k = 1:size(identical)[1]
                        if(data[j,i] == identical[k])
                            data[j,i] = Int(k)
                        end
                end
            end
        end
    end

#convertiing from array of any to array of float64
d = convert(Array{Float64},data[:,:])
    return d
end

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

#function for calculating predicting y
function hypothesis(z)
    return sigmoid(z)
end


# cost function for computing the cost of all taining sample
function regularised_cost_function(estimate,y,theta,lambda)
    m = length(y)
    total_cost = ((-y)' * log.(estimate))-((1 .- y)' * log.(1 .- estimate)) .+  (lambda/(2*m) *  broadcast(abs, theta))
    return  total_cost
end


#fuction for calculating the derivative 
function derivative(estimate, y, X)
       derivative_of_cost = (estimate - y)*X
       return derivative_of_cost
end


#function for calculating  gradient descend
function gradient_descent(X,y,lambda,learning_rate)
    length=size(X)[2]
    #initialising theta with zeros
    theta = zeros(Int, 1, size(X, 2))
    
    #calculating gradient and updating theta
    for i in 1:(size(X)[1])
        gradient = derivative(hypothesis(theta*X[i, :]), y[i], X[i, :] )
        theta = theta .- (learning_rate * gradient) .+ lambda* broadcast(abs, theta)
    end
    
    return theta
end


# #function for making prediction on unseen data
function predict(xx)
    #getting the updated theta from gradient descent
    theta  = gradient_descent(X_train,y_train,0.001,10)

    #intialising the probability array to hold the calculated estimate
    probability = zeros(size(xx)[1])
    
    #multiplying the X-test sample with updated theta
    for i in 1:(size(xx)[1])
        probability[i]= hypothesis(theta*X[i, :])
        
        #deciding the threshold
        if(probability[i]< 0.5)
            probability[i]=0
            else
                probability[i]=1
            end
    end

    return probability
end


# #confusion matrix (accurace,pecision,recall)
function confusion_matrix(estimate , actual)
    #initialising the variable for using to calculate CM
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

       #getting TP and FN
        if(estimate[i]== 0 )
            if(actual[i]== estimate[i])
            true_positive = true_positive +1 
            elseif(actual[i] != estimate[i])
            false_negative = false_negative +1
            end
        end

        #getting TN and FP
        if(estimate[i]== 1 )
            if(actual[i]== estimate[i])
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


#envoking the methoin predict 
prediction = predict(X_test)


#envoking the method confusion matrix
accurace ,recall ,precision = confusion_matrix(prediction,y_test)

print("\nrecall :",recall)
print("\naccurace :",accurace)
print("\nprecision  :",precision)
print("\n")
