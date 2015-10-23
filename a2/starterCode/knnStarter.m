
% Use the function knnClassify to test performance on different datasets.
figure(1);clf;


load('a2/dataSets/generic1');
n1t = size(c1_train, 2);
n2t = size(c2_train, 2);
trainingInputs = [c1_train, c2_train];
trainingTargets = [ones(1, n1t) zeros(1, n2t); zeros(1, n1t) ones(1, n2t)];

n1 = size(c1_test, 2);
n2 = size(c2_test, 2);
testInputs = [c1_test, c2_test];
testTargets = [ones(1, n1) zeros(1, n2); zeros(1, n1) ones(1, n2)];

E = [];
for k = 3:2:21

    num_error = 0;

    for i = 1:(n1+n2)
        class = knnClassify(testInputs(:, i)', k, trainingInputs', trainingTargets');
        if class ~= testTargets(:, i)'
            num_error = num_error + 1;
        end
    end
    E = [E; k num_error/(n1+n2)];
end

plot(E(:, 1), E(:, 2), '--ms');
hold on;
% ==============================================================

load('a2/dataSets/generic2');

n1t = size(c1_train, 2);
n2t = size(c2_train, 2);
trainingInputs = [c1_train, c2_train];
trainingTargets = [ones(1, n1t) zeros(1, n2t); zeros(1, n1t) ones(1, n2t)];

n1 = size(c1_test, 2);
n2 = size(c2_test, 2);
testInputs = [c1_test, c2_test];
testTargets = [ones(1, n1) zeros(1, n2); zeros(1, n1) ones(1, n2)];

E = [];
for k = 3:2:21

    num_error = 0;

    for i = 1:(n1+n2)
        class = knnClassify(testInputs(:, i)', k, trainingInputs', trainingTargets');
        if class ~= testTargets(:, i)'
            num_error = num_error + 1;
        end
    end
    E = [E; k num_error/(n1+n2)];
end

plot(E(:, 1), E(:, 2), '--bs');
hold on;

% ==============================================================

load('a2/dataSets/fruit_train');
load('a2/dataSets/fruit_test');

E = [];
for k = 3:2:21
    
    num_error = 0;

    for i = 1:size(inputs_test, 2)
        class = knnClassify(inputs_test(:, i)', k, inputs_train', target_train');
        if class ~= target_test(:, i)'
            num_error = num_error + 1;
        end
    end
    E = [E; k num_error/size(inputs_test, 2)];
end

plot(E(:, 1), E(:, 2), '--go');
hold on;

% ==============================================================

load('a2/dataSets/mnist_train');
load('a2/dataSets/mnist_test');

E = [];
for k = 3:2:21
    
    num_error = 0;

    for i = 1:size(inputs_test, 2)
        class = knnClassify(inputs_test(:, i)', k, inputs_train', target_train');
        if class ~= target_test(:, i)'
            num_error = num_error + 1;
        end
    end
    E = [E; k num_error/size(inputs_test, 2)];
end

plot(E(:, 1), E(:, 2), '--yo');

legend('generic1', 'generic2', 'fruits', 'digits');
xlabel('k');
ylabel('test error');
title('Test Error of KNN as a function of k');