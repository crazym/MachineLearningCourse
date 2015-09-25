load('a1TrainingData.mat');
load('a1TestData.mat');

figure(1);
% random color map (12-by-3 vector)
cmap = hsv(12);
for K = 1:12
    w = polynomialRegression(K, x, y)
    E = norm(y - evalPolynomial(x, w))
    errorArray(K) = E;
    testE = norm(yTest - evalPolynomial(xTest, w))
    newErrorArray(K) = testE;
    
    my_x = -2.1:0.1:2.1;
    % plot data set regarding each degree K in a randome color in cmap
    plot(my_x, evalPolynomial(my_x, w), 'Color', cmap(K, :));
    % make a cell array with legend info so we can call legend after the loop
    legendInfo{K} = ['K = ' num2str(K)];
    hold on;
end
legend(legendInfo)
xlabel('x')
ylabel('y')
title('Fitted model for x = [-2.1:0.1:2.1] in different K')

% error plot on training data
figure(2);
plot(1:12, errorArray, '--bs');
xlabel('degree K');
ylabel('Total Error E = norm(y-evalPolynomial(x, w)');
title('Total Residual error on Training Data as a function of K');

% total error plot test data
figure(3);
plot(1:12, newErrorArray, '--bs');
xlabel('degree K');
ylabel('Total Error E = norm(y-evalPolynomial(x, w)');
title('Total Residual error on TestData as a function of K');