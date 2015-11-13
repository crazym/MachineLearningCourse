function [P, a_1, a_0] = navie(X, y, data_train, labels_train, alpha, beta)
  % regularized form of Navie Baye's classification on C=1
  % X is a M x N matrix, columns are data vectors
  % y is a 1 x N vector, each labels the class of the corresponding data
  % alpha and beta are regularization parameters
  
  % P is a 1 x N vector where each p represents the classified class 
  % i.e. for each p in P:
  % p(C=1|F_{1, 185}) = exp (a_1 - r) / (exp (a_1 - r) + exp (a_0 - r))
  % where:
  % a_k = sum_{i:F_i=1}(ln a_{i,k}) + sum_{i:F_i=0}(ln 1 - a_{k,1}) + ln b_k
  % for k = 0, 1
  %
  % and:
  % a_{i, k} = (sum(F_i=k) + alpha)/ (sum(label=k) + 2*alpha)
  % b_1 = (sum(label=1) + beta) / (N + 2*beta)
  % b_0 = 1 - b_1
  % r = max {a_0, a_1}
  
  M = size(data_train, 1);
  N = size(labels_train, 2);    
  num_c1 = sum(labels_train);
  num_c0 = N - num_c1;
  
  % columns of data of different class
  c0_train = data_train(:, find(labels_train==0));
  c1_train = data_train(:, find(labels_train==1));
  % a_{i, 1} = (sum(F_i=1) + alpha)/ (num_C1 + 2*alpha)
  a_1 = zeros(0, M);
  a_0 = zeros(0, M);
  for i = 1:M
    a_1(i) = (sum(c1_train(i, :)) + alpha) / (num_c1 + 2*alpha);
    a_0(i) = (sum(c0_train(i, :)) + alpha) / (num_c0 + 2*alpha);
  end
  
  b_1 = (num_c1 + beta) / (N + 2*beta);
  b_0 = 1 - b_1;
  
  P = zeros(0, size(y, 2));
  for n = 1:size(y, 2)
      a0 = 0;
      a1 = 0;
      for m = 1:M
          if X(m, n) == 1
              a1 = a1 + log(a_1(m));
              a0 = a0 + log(a_0(m));
          else
              a1 = a1 + log(1 - a_1(m));
              a0 = a0 + log(1 - a_0(m));              
          end
      end
      a1 = a1 + log(b_1);
      a0 = a0 + log(b_0);
      r = max(a1, a0);
      
      P(n) = exp(a1 - r) / (exp(a1 - r) + exp(a0 - r));
  end
      
  return;
