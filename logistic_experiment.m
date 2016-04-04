function [weights, accuracy, X, Y] = logistic_experiment(n)
% Logistic regression experiment:
% Divide the data set into two part and fit the logistic regression model
% with the firs n rows of trainig set. Then apply the model to the testing 
% set and return the accuracy of the model in the testing set.
% 
% Input: 
%         n   the first n rows of the data set;
%         
% Output:
%         The accuracy of the model in the testing data set. The accuracy
%         of the model is represented by the value of AUC.
%         The weights of the logisitc regression model
%         True positive rate and False positive rate.
%        

data_Spam = textread('data_Spam.txt');
labels_Spam = textread('labels_Spam.txt');

% Creating the training and testing data sets
a = size(data_Spam, 1);
temp_intecp = ones(a, 1);
data_Spam = [data_Spam, temp_intecp,];
train_Spam = data_Spam(1:2000, :);
train_labels = labels_Spam(1: 2000, :);
test_Spam = data_Spam(2001:4601, :);
test_labels = labels_Spam(2001:4601, :);


temp_data = train_Spam(1 : n, : );
temp_labels = train_labels(1 : n);
weights = logistic_train(temp_data, temp_labels);

P = zeros(1, size(test_Spam, 1));

for i = 1 : size(test_Spam, 1)
 P(i)= 1/ (1+ exp(- test_Spam(i,:) * weights) ); % denotes the probability
end
scores = P';
[X, Y, T, accuracy] = perfcurve(test_labels, scores, '1');




    