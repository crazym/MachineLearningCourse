% Use the function learnGCCmodel to learn a Gaussian Class-Conditional
% model for classification.
% Use the function gccClassify for evaluating the decision function
% to perform classification.
% Then test the model under various conditions.
figure(1);clf;

load('a2/dataSets/generic1');


x1 = c1_train';
x2 = c2_train';

[p1, m1, m2, C1, C2] = learnGCCmodel(x1, x2);

for i = 1:size(c1_test, 2)
    class = gccClassify(c1_test(:, i), p1, m1, m2, C1, C2);
    plot(c1_test(1, i),c1_test(2, i), 'bx');hold on;
    if class ~= [1 0]
        plot(c1_test(1, i),c1_test(2, i), 'ro', 'MarkerSize', 5);
    end
end

for i = 1:size(c2_test, 2)
    class = gccClassify(c2_test(:, i), p1, m1, m2, C1, C2)
    plot(c2_test(1, i),c2_test(2, i), 'go');hold on;
    if class ~= [0 1]
        plot(c2_test(1, i),c2_test(2, i), 'rs', 'MarkerSize', 5);
    end
end

xlabel('x');
ylabel('y');
legend('C1','C2');
title('GCC fitting generic1');

==============================================================
figure(2);clf;
load('a2/dataSets/generic2');


x1 = c1_train';
x2 = c2_train';

[p1, m1, m2, C1, C2] = learnGCCmodel(x1, x2);

for i = 1:size(c1_test, 2)
    class = gccClassify(c1_test(:, i), p1, m1, m2, C1, C2);
    plot(c1_test(1, i),c1_test(2, i), 'bx');hold on;
    if class ~= [1 0]
        plot(c1_test(1, i),c1_test(2, i), 'ro', 'MarkerSize', 5);
    end
end

for i = 1:size(c2_test, 2)
    class = gccClassify(c2_test(:, i), p1, m1, m2, C1, C2)
    plot(c2_test(1, i),c2_test(2, i), 'go');hold on;
    if class ~= [0 1]
        plot(c2_test(1, i),c2_test(2, i), 'rs', 'MarkerSize', 5);
    end
end

xlabel('x');
ylabel('y');
legend('C1','C2');
title('GCC fitting generic2');

% ==============================================================

figure(3);clf;

load('a2/dataSets/fruit_train');
load('a2/dataSets/fruit_test');

c1 = find(target_train(1,:)==1);
c2 = find(target_train(2,:)==1);
x1 = inputs_train(:,c1);
x2 = inputs_train(:,c2);

[p1, m1, m2, C1, C2] = learnGCCmodel(x1', x2');

for i = 1:size(inputs_test, 2)
    class = gccClassify(inputs_test(:, i), p1, m1, m2, C1, C2)
    if class == [0 1]
        plot(inputs_test(1, i),inputs_test(2, i), 'bx');hold on;
    else
        plot(inputs_test(1, i),inputs_test(2, i), 'gs');hold on;
    end
   
    if class ~= target_test(:, i)'
        plot(inputs_test(1, i),inputs_test(2, i), 'ro', 'MarkerSize', 5);
    end
end

xlabel('x');
ylabel('y');
legend('C1','C2');
title('GCC fitting fruits');