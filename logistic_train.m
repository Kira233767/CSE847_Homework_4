function [weights] = logistic_train(data, labels, epsilon, maxiter)
% 
% Code for train a logistic regression classifier
% 
% Inputs:
%     data = n * (d+1) matrix with n samples and d features, where column 
%              d+1 is all ones
%            (corresponding to the intercept term)
%     
%     labels = n*1 vector of class labels (taking values 0 or 1)
%     
%     epsilon = optional argument specifying the covergence criterion - if 
%                 the cahnge in the absolute difference in predictions, 
%                 form one iteration to the next, averaged across input 
%                 features, is less than epsilon, then halt
%               (if unspecified, use a default value of 1e-5)
%         
%     maxiter = optional argument that specifies the maximum number of 
%                 iterations to execute (useful when debugging in case your 
%                 code is not voncerging correctly)
%               (if unspecified, can be set to 1000)
%         
%  Output:
%     weights = (d+1) * 1 vector of weights where the weights correspond to
%                 the columns of data

% Check input
if nargin < 4
    maxiter = 1000;
    if nargin < 3
        epsilon = 1e-5;
        if nargin < 2
            error(message('logistic_train does not have enough input!'));
        end
    end
end


phi = data;                 %design matrix (intercet term are included)
t = labels;                 %Target variables
n = size(phi, 2);

w_new = zeros(n, 1);            %initial values of weight
w = ones(n, 1);
counts = 0;



while(abs(sum(w_new - w)) > epsilon & counts <= maxiter)
   w = w_new;
   z = phi * w;
   angle = zeros(1, length(z));
   for i = 1 : length(z)
       angle(i) = 1/ (1 + exp(-z(i)));
   end
   R = diag(angle);
   w_new = (phi' * R * phi)^(-1) * phi' * (R * phi * w - (z - t));
   counts = counts + 1;
    
end

weights = w_new;

