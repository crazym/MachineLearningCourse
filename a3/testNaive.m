load('a3spam.mat')

figure(1);clf;
N_test = size(labels_test, 1);
N_train = size(labels_train, 1);
error_naive_train = zeros(1, 9);
error_naive_test = zeros(1, 9);
% WLOG we assume alpha == beta for simplicity
i = 1;
for regulator = 0.1:0.05:0.5
    [P_c1_train, a_1, a_0] = navie(data_train', labels_train', data_train', labels_train', regulator, regulator);
    classes_train = P_c1_train > 0.5;
    error_naive_train(i) = (N_train - sum(classes_train==labels_train'));
    
    P_c1_test = navie(data_test', labels_test', data_train', labels_train', regulator, regulator);
    classes_test = P_c1_test > 0.5;
    error_naive_test(i) = (N_test - sum(classes_test==labels_test'));
    i = i + 1;
end

% plot(0.1:0.05:0.5, error_naive_train, '--go');
% hold on;
plot(0.1:0.05:0.5, error_naive_test, '--bs');

% legend('training data', 'test data');
xlabel('regulator: alpha(==beta)');
ylabel('test error');
title('Test Error of Naive Bayes as a function of alpha==beta');


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

% choose alpha = beta = 0.1
regulator = 0.1;
[P_c1_test, a_1, a_0] = navie(data_test', labels_test', data_train', labels_train', regulator, regulator);
classes_test = P_c1_test > 0.5;

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
top_spam_weights = a_1(top_spam_index)

diff_freq(index(176:185));
% features appears more often in C0 than C1:
top_ham_index = fliplr(index(176:185));
top_ham = feature_names(top_ham_index)'
top_ham_weights = a_0(top_ham_index)





