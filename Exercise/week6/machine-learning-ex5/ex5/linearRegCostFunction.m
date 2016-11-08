function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



% this is Linear Regression, not logistic regression

#{
grad = 1/m*X'*(sigmoid(X*theta)-y);
temp = theta;
temp(1) = 0;
grad = grad + lambda/m*temp;

J = 1/m*(-(log(sigmoid(X*theta)))'*y-(log(1-sigmoid(X*theta)))'*(1-y))+lambda/2/m*sum(temp.^2);

#}

grad = 1/m*X'*(X*theta-y);
temp = theta;
temp(1) = 0;
grad = grad + lambda/m*temp;

J = 1/2/m*sum(((X*theta)-y).^2)+lambda/2/m*sum(temp.^2);



% =========================================================================

grad = grad(:);

end
