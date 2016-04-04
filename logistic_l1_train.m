function [w, c, AUC, X, Y] = logistic_l1_train(data1, data2, labels1,labels2, par)
% OUTPUT w is equivalent to the first d dimension of weights in logistic train
% c is the bias term, equivalent to the last dimension in weights in logistic train.
% Specify the options (use without modification).
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations



[w, c] = LogisticR(data1, labels1, par, opts);

% scores are used to compute the probability predicted by the logistic 
% regression model
% compute the score in the function so that I can also apply perfcurve
% function and get the AUC values

P = zeros(1, size(data2, 1));

for i = 1 : size(data2, 1)
 P(i)= 1/ (1+ exp(- (data2(i,:) * w + c)) ); % denotes the probability
end
[X, Y, T, AUC] = perfcurve(labels2, P', '1');

end
