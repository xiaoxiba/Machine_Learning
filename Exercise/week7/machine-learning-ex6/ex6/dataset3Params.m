function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

% value find by training and verify on validation set
C = 1;
sigma = 0.1;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

#{
C_list = [0.01,0.03,0.1,0.3,1,3,10,30,100];
sigma_list = [0.01,0.03,0.1,0.3,1,3,10,30,100];

#C_list = [0.01];
#sigma_list = [0.01, 0.1];

error_train = zeros(length(C_list), length(sigma_list));
error_val = zeros(length(C_list), length(sigma_list));


for i = 1:length(C_list)
  
  C_var = C_list(i);
  
  for j = 1:length(sigma_list)
  
    sigma_var = sigma_list(j);
 
    % train model
    model= svmTrain(X, y, C_var, @(x1, x2) gaussianKernel(x1, x2, sigma_var));
    
    % calculate train and validation error
    p_train = svmPredict(model, X);
    p_val = svmPredict(model, Xval);
    
    % calculate train and validation error
    error_train(i,j) = mean(double(p_train ~=y));
    error_val(i,j) = mean(double(p_val ~=yval));
    
  endfor
endfor
 
    



[i j] = find(error_val == min(min(error_val)));

C = C_list(i(1));
sigma = sigma_list(j(1));

save -mat dataset3Params_final.mat error_train error_val C sigma;

#}

% =========================================================================

end
