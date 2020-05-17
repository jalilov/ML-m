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

hypothesis =  X * theta;
J = 1/(2*m) * sum( (hypothesis-y).^2 ) + (lambda/(2*m) * sum(theta(2:end).^2));

% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
% 
grad =  1/m * sum( (hypothesis - y) .*X, 1); % sum over dimension 1 is important, otherwise sums over the fist row when there is a single element in each column 
%derivative of the gradient descent
% derivative  =  1/m * sum((hypothesis - y).*X);
grad(:,2:end) =  grad(1,2:end) + lambda/m*theta(2:end)';%regularize only the second gradient; not for Q0

% =========================================================================

grad = grad(:);

end
