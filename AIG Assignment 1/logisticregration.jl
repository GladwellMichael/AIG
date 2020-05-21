using CSV
using Statistics
using  PyCall


#reading dataset for experimenting purpose
df =CSV.read("ex2data1.csv")

#converting dataframe to matrix
df = convert(Matrix,df)

# function for standadising the data
function standadiation()
    
end

#function for cleaning data
function datacleaning(data)
    #code to come here
    return data
end


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
# function cost_function(features,y,theta)
#     m,n = size(features)
#     total_cost =(-(y*log(hypothesis(features,theta))+(1-y)log(1-hypothesis(X,theta)))
#     return  total_cost
# end

function derivative(predicted_y,actual,feature)
    return (predicted_y - actual)*feature
end


# # computing the gradient of cost function
function gradient_descent(features,y,rate)
    m ,n = size(features)
    theta = zeros(1,2)
    for i in 1:m
        theta  = theta-transpose(derivative(hypothesis(theta*features[i,:]),y[i],features[i,:]))
    end
    return theta
    # gradient =  (1/m)*((hypothesis(features,theta))^2) # y is missing and was causing some error(dimension issues)
    # print("gradient :",gradient ,"\n")
    # return  gradient
end


gradient_descent(X_train,y_train,0.1)


model = gradient_descent(X_train,y_train,0.1)

# predict function for making predictions on our model
function predict(X)
    theta = model[:]
    print("\nTheta",theta)
    print("\n X",X)
    m,n = size(X)
    result = 0
    correctly_predicted =0
    for i in 1:m
        result = hypothesis(theta.*X[i,:])
        x_test = X_test[i, :]
        results = y_test[i]
        if(hypothesis(theta .*x_test) == results)
            correctly_predicted += 1
        end
    end
    print("\n result",result)
    print("The percentage of the correct is " ,correctly_predicted/m*100 )
    return result
end
prediction = predict(X_test)
print("\nPredictions",prediction)

# #Testing the perfomance of the model using the confusing matrix(accuracy,precison,recall)
# function accuracy(X,y_test,probability_threshold=0.5)
#     predicted_classes = (predict(X) >= probability_threshold)
#     predicted_classes = predicted_classes.flatten()
#     accuracy = mean(predicted_classes == y_test)
#     return accuracy * 100
    
# end 
    
# predict = predict(X_test)
# print(predict)