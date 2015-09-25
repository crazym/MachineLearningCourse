load('a1TrainingData.mat');
figure(1);
cmap = hsv(12);
for K = 1:12
    w = polynomialRegression(K, x, y)
    E = norm(y - evalPolynomial(x, w))
    errorArray(K) = E
    my_x = -2.1:0.1:2.1;
    plot(my_x, evalPolynomial(my_x, w), 'Color', cmap(K, :));
%   make a cell array with legend info so we can call legend after the loop
    legendInfo{K} = ['K = ' num2str(K)];
    hold on;
end
legend(legendInfo)
title('Fitted model for x = [-2.1:0.1:2.1] in different K')

figure(2)
plot(1:12, errorArray(K))
title('Total Residual error as a function of K')