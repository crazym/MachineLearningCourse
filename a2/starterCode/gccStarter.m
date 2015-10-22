% Use the function learnGCCmodel to learn a Gaussian Class-Conditional
% model for classification.
% Use the function gccClassify for evaluating the decision function
% to perform classification.
% Then test the model under various conditions.


load('a2/dataSets/generic1');


x1 = c1_train';
x2 = c2_train';

[p1, m1, m2, C1, C2] = learnGCCmodel(x1, x2);

for i = 1:size(c1_test, 2)
%     c1_test(:, i)
    gccClassify(c1_test(:, i), p1, m1, m2, C1, C2)
end

for i = 1:size(c2_test, 2)
%     c1_test(:, i)
    gccClassify(c2_test(:, i), p1, m1, m2, C1, C2)
end