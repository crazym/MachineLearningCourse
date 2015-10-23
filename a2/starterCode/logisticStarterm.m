% student number: 999586755
% Use learnLogReg() to test performance on various datasets.
% =============================================================
% Fruits
load('fruit_train');
load('fruit_test');

alphas = [0.25, 0.5, 1, 2, 5];

count_alphas = size(alphas,2);
count_tests = size(target_test,2);
test_error = zeros(1, count_alphas);
data_c1 = find(target_train(1,:)==1);
data_c2 = find(target_train(2,:)==1);
train_c1 = inputs_train(:,data_c1);
train_c2 = inputs_train(:,data_c2);


for i = 1:count_alphas
    error = 0;
    [w, b] = learnLogReg(train_c1, train_c2, alphas(i));
    guess = logistic(inputs_test, w, b);
    for j = 1:count_tests
        if (round(guess(j)) ~= target_test(1, j))
            error = error + 1;
        end
    end
    test_error(1,i) = error/count_tests;
end

figure(1);
plot(alphas, test_error, '--mo');
hold on;

% =============================================================
% Digits

load('mnist_train');
load('mnist_test');


count_alphas = size(alphas,2);
count_tests = size(target_test,2);
test_error = zeros(1, count_alphas);
data_c1 = find(target_train(1,:)==1);
data_c2 = find(target_train(2,:)==1);
train_c1 = inputs_train(:,data_c1);
train_c2 = inputs_train(:,data_c2);

for i = 1:count_alphas
    error = 0;
    [w, b] = learnLogReg(train_c1, train_c2, alphas(i));
    guess = logistic(inputs_test, w, b);
    for j = 1:count_tests
        if (round(guess(j)) ~= target_test(1, j))
            error = error + 1;
        end
    end
    test_error(1,i) = error/count_tests;
end

plot(alphas, test_error, '--bo');
hold on;

% =============================================================
% Generic1
load('generic1');

count_alphas = size(alphas,2);
test_error = zeros(1, count_alphas);
g1_inputs_test = [c1_test c2_test];
g1_target_test = [diag([1,0])*ones(size(c1_test)) diag([0 1])*ones(size(c2_test))];
count_tests = size(g1_target_test,2);


for i = 1:count_alphas
    error = 0;
    [w, b] = learnLogReg(c1_train,c2_train, alphas(i));
    guess = logistic(g1_inputs_test, w, b);
    for j = 1:count_tests
        if (round(guess(j)) ~= g1_target_test(1, j))
            error = error + 1;
        end
    end
    test_error(1,i) = error/count_tests;
end
plot(alphas, test_error,'--ro'); hold on;
% =============================================================
% Generic2
load('generic2');

count_alphas = size(alphas,2);

test_error = zeros(1, count_alphas);
g2_inputs_test = [c1_test c2_test];
g2_target_test = [diag([1,0])*ones(size(c1_test)) diag([0 1])*ones(size(c2_test))];
count_tests = size(g2_target_test,2);

for i = 1:count_alphas
    error = 0;
    [w, b] = learnLogReg(c1_train,c2_train, alphas(i));
    guess = logistic(g2_inputs_test, w, b);
    for j = 1:count_tests
        if (round(guess(j)) ~= g2_target_test(1, j))
            error = error + 1;
        end
    end
    test_error(1,i) = error/count_tests;
end
plot(alphas, test_error,'--go'); hold on;

xlabel('Alpha');
ylabel('Test Error Rate');
legend('Fruit','Digits','Generic1','Generic2');
title('Logistic Regression Test');
