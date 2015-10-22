
function class = gccClassify(x, p1, m1, m2, C1, C2)
% 
% Inputs
%   x - test examplar
%   p1 - prior probability for class 1
%   m1 - mean of Gaussian measurement likelihood for class 1
%   m2 - mean of Gaussian measurement likelihood for class 2
%   C1 - covariance of Gaussian measurement likelihood for class 1
%   C2 - covariance of Gaussian measurement likelihood for class 2
%
% Outputs
%   class - sgn(a(x)) (ie sign of decision function a(x))



% YOUR CODE GOES HERE.
a = - (1/2) * transpose(x - m1) * inv(C1) * (x - m1) - (1/2) * ln (norm(C1)) ...
    + (1/2) * transpose(x - m2) * inv(C2) * (x - m2) - (1/2) * ln (norm(C2));

if a > 0
    class = [1 0];
else
    class = [0 1];
end
end