
% Use the function knnClassify to test performance on different datasets.


load('a2/dataSets/generic1');

k = 2;

x1 = c1_train';
x2 = c2_train';


E = [];
for k = 3:2:21
    
    n1 = size(c1_test, 2);
    n2 = size(c2_test, 2);
    
    num_error = 0;
    trainingInputs = [c1_test, c2_test];
    trainingTargets = [ones(1, n1) zeros(1, n2); zeros(1, n1) ones(1, n2)];
    
    for i = 1:(n1+n2)
    %     c1_test(:, i)
        class = knnClassify(trainingInputs(:, i)', k, trainingInputs', trainingTargets');
        if class ~= trainingTargets(:, i)'
            num_error = num_error + 1;
        end
    end
    E = [E; k num_error/(n1+n2)];
end
