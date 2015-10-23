function class = knnClassify(test, k, trainingInputs, trainingTargets)
%
% Inputs:
%   test: test input vector as a row vector
%   k: number of nearest neighbours to use in classification.
%   traingingInputs: array of training exemplars, one exemplar per row
%   traingingTargets: idenicator vector per row
%
% Basic Algorithm of kNN Classification
% 1) find distance from test input to each training exemplar,
% 2) sort distances
% 3) take smallest k distances, and use the median class among 
%    those exemplars to label the test input.
%


% YOUR CODE GOES HERE.
n = size(trainingInputs, 1);

% allocates matrix to store distances and their indexes
D = ones(n, 2);

for i  = 1:n
    distance = sum((trainingInputs(i, :) - test) .^ 2)^(1/2);
    D(i, :) = [distance i];
end

sorted = sortrows(D);

nearestMedian = sorted(floor(k/2), 2);

class = trainingTargets(nearestMedian, :);



    
