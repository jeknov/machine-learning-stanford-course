function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

C=((X*Theta')-Y).*R; %calculates the part, common for all the calculations

J=(1/2)*sum(sum((C).^2)); % calculates the unregularized cost function value

X_grad=C*Theta; % calculates the unregularized X gradient
Theta_grad=C'*X; % calculates the unregularized Theta gradient

temp1=Theta;
% temp1(1)=0; % not necessary for this algorithm
temp2=X;
% temp2(1)=0; % not necessary for this algorithm

J=J+(lambda/2)*sum(sum(Theta.^2))+(lambda/2)*sum(sum(X.^2)); % calculates the regularized cost function value
Theta_grad=Theta_grad+(lambda*temp1); % calculates the regularized Theta gradient
X_grad=X_grad+(lambda*temp2);  % calculates the regularized X gradient

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
