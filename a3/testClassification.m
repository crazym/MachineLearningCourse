load('a3spam.mat')
res = navie(data_test', labels_test', data_train', labels_train', 0.5, 0.5) > 0.5;

error = 4000 - sum(res==labels_test')