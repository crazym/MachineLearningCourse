load('a3spam.mat')

figure(1);clf;
N = size(labels_test, 1);
error_lr = zeros(1, 9);

X = [ones(1, 1000); data_train'];
i = 1;
for v = 1:0.5:5
    [beta, converged] = logisticReg(X, labels_train', v);
    P_c1 = logistic([ones(1, 4000); data_test'], beta);
    classes = P_c1 > 0.5;
    error_lr(i) = N - sum(classes==labels_test');
    i = i + 1;
end
plot(1:0.5:5, error_lr, '--bo');

xlabel('regulator: v (variance for a Gaussian prior on the weights)');
ylabel('test error');
title('Test Error of Logistic Regression as a function of weights');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Top 10 features
% Idea:
%   separate the test data set according to the classification result,
% count the occurences frequencies of each feature in both classes.
%
% However, we noticed that there are words apprears to be common in both
% classes, which does not act as indicative, so instead we define the
% "indicativeness" of a feature to be of high difference in frequencies in
% two classes.

% choose v = 4
v=4;
[beta, converged] = logisticReg(X, labels_train', v);
P_c1 = logistic([ones(1, 4000); data_test'], beta);
classes = P_c1 > 0.5;
% of size n x 185
c0 = data_test(find(classes_test==0), :);
c1 = data_test(find(classes_test==1), :);

% count occurrences of each feature in both classes
% of size 1 x 185
occur_c0 = sum(c0);
occur_c1 = sum(c1);

% frequencies occurrences / number of entries
freq_c0 = occur_c0 ./ size(c0, 1);
freq_c1 = occur_c1 ./ size(c1, 1);

diff_freq = freq_c0 - freq_c1;

[~, index] = sort(diff_freq);
diff_freq(index(1:10));
% based on the context, we assume label==1 is spam
% features appears more often in C1 than C0:
top_spam_index = index(1:10);
top_spam = feature_names(top_spam_index)'
top_spam_weights = beta(top_spam_index)'

diff_freq(index(176:185));
% features appears more often in C0 than C1:
top_ham_index = fliplr(index(176:185));
top_ham = feature_names(top_ham_index)'
top_ham_weights = beta(top_ham_index)'
