function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% The following variables are returned:
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

v=x1-x2; % calculating the distance/similarity between x1 and x2
v=v.^2; % calculating the square distance value

sim=exp(-sum(v)/(2*(sigma^2))); % calculating the Gaussian kernel


% =============================================================
    
end
