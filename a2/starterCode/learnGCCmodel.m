function [p1, m1, m2, C1, C2] = learnGCCmodel(x1, x2)
% 
% Inputs
%   x1 - training exemplars from class 1, one exemplar per row
%   x2 - training exemplars from class 2, one exemplar per row
%
% Outputs
%   p1 - prior probability for class 1
%   m1 - mean of Gaussian measurement likelihood for class 1
%   m2 - mean of Gaussian measurement likelihood for class 2
%   C1 - covariance of Gaussian measurement likelihood for class 1
%   C2 - covariance of Gaussian measurement likelihood for class 2
%


