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

% number of training examplars
n1 = size(x1, 2);
n2 = size(x2, 2);

% do necessary transposes in order to get mean
m1 = sum(x1')' / n1;
m2 = sum(x2')' / n2;

C1 = 0;
C2 = 0;
% for i = 1:n1
%     diff = x1(:, i) - m1;
%     C1 = C1 + diff * diff';
% end

diff1 = x1 - m1*ones(1, n1);
C1 = (1 / n1) * (diff1)' * diff1;

C2 = cov(x2', 1);
% 
% for i = 1:n2
%     diff = x2(:, i) - m2;
%     C2 = C2 + diff * diff';
% end
% 
% C1 = C1 ./ n1;
% C2 = C2 ./ n2;

end