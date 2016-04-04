% This is the main function for homework 4. It use logistic regression 
% functions to fit the model and get the AUC values as accuracy. 

% Logisitc regression experiment:
clear; clc;

n = [200, 500, 800, 1000, 1500, 2000];
Acc1 = zeros(1, 6);
for i = 1 : 6
    [a,b,c,d] = logistic_experiment(n(i));
    Acc1(i) = b;
end

plot(n, Acc1)  %AUC against traing set size plot
xlabel('Size of training set')
ylabel('AUC values/ accuracy')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sparse logistic regression experiment
clear; clc;
load('ad_data.mat');


%regularization parameters
par = 1e-8 : 0.01 : 1;
k = length(par);
feature_num = zeros(length(par));
Acc2 = zeros(1, k);
for i = 1 : k
    [a, b, c, d, e] = logistic_l1_train(X_train, X_test,...
    y_train, y_test, par(i));
    feature_num(i) = length(find(a ~= 0));
    Acc2(i) = c;
end

subplot(1, 2, 1)
plot(par, feature_num)
xlabel('Regularization parameters')
ylabel('Number of features')

subplot(1, 2, 2)
plot(par, Acc2)
xlabel('Regularization parameters')
ylabel('AUC values/ Accuracy')