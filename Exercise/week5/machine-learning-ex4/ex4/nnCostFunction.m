function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
               
                 
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1 :Feedforward the neural network and return the cost in the
%         variable J.

% Add ones to the X data matrix

X = [ones(m, 1) X];


y_label=zeros(length(y));

% calculate output layer activiation with theta values
  
    % get layer2 (hidden layer) activiation
    z2 = X*Theta1';
    a2 = sigmoid(z2);
    % Add ones to the a2 matrix
    a2 = [ones(m, 1) a2];
    % get output layer a3 activiation
    z3 = a2*Theta2';
    a3 = sigmoid(z3);
    
    
% calculate cost for each of 10 classes labels

for i = 1:num_labels
 
  %First get y_label and a3_label for one-vs-all

  y_label = (y==i);
  a3_label = a3(:,i);
  
  % calculate J(theta) for i th class
  J = J + 1/m*(-(log(a3_label))'*y_label-(log(1-a3_label))'*(1-y_label));
  
  
endfor


% Part 2: Implement the backpropagation algorithm to compute the gradients

% Step 1:
% calculate output layer activiation with theta values 
% this calculation is duplicated with Part 1 calculation, put here twice just for reference, remove it when training
  
    % get layer2 (hidden layer) activiation
    z2 = X*Theta1';
    a2 = sigmoid(z2);
    % Add ones to the a2 matrix
    a2 = [ones(m, 1) a2];
    % get output layer a3 activiation
    z3 = a2*Theta2';
    a3 = sigmoid(z3);
    
% Step 2: For each output unit k in layer 3 (the output layer), set

delta3 = zeros(size(X,1),num_labels);

for i = 1:num_labels
 
  %First get y_label and a3_label for one-vs-all

  y_label = (y==i);
  a3_label = a3(:,i);
  
  % calculate error for i th class
  delta3_label = a3_label-y_label;
  
  % append i th delta3 to whole delta3 matrix
  delta3(:,i) = delta3_label;
    
endfor


% Step 3: For the hidden layer l = 2, set

% add 1 to z2
z2 = [ones(m, 1), z2];
%calculate delta2
delta2 = (delta3*Theta2).*(sigmoidGradient(z2));


%Step 4: Accumulate the gradient from this example


% remove delta2(0)
delta2 = delta2(:,2:end);

Delta_2 = a2'*delta3;
Delta_1 = X'*delta2; % remove the x0 bias


%Step 5: Obtain the (unregularized) gradient for the neural network cost func-tion by dividing the accumulated gradients by 1/m

Delta_2 = 1/m*Delta_2;
Delta_1 = 1/m*Delta_1;

% calculate gradient
Theta1_grad = Delta_1';
Theta2_grad = Delta_2';






% Part 3: Implement regularization with the cost function and gradients.

  Theta1_temp = Theta1;
  Theta2_temp = Theta2;

  % first theta parameter is not regulaterized, so set it temperally to 0
  Theta1_temp(:,1) = 0;
  Theta2_temp(:,1) = 0;

  % cost function
  J = J + lambda/2/m*(sum(sum(Theta1_temp.^2))+sum(sum(Theta2_temp.^2)));

  % grad wit regularization
  Theta1_grad = Theta1_grad +lambda/m*Theta1_temp;
  Theta2_grad = Theta2_grad +lambda/m*Theta2_temp;
  
  
  
  
  
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
