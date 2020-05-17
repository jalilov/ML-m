function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters,2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
  % ======================================='=====================  
    %Vectorized method
    h = X * theta; %hypothesis 
    errors_vec = h-y; %difference between the hypothesis and actual values of y
    % calculate the change in theta for the next step
    theta_change = alpha * (1/m) * (X' * errors_vec);
    theta = theta - theta_change;
    % Save the cost J in every iteration    
    J_history(iter,1:2) = computeCost(X, y, theta);
    
end

end
