function p = logistic(X, w, b)
  % Evaluate logistic function
  % X is a M x N matrix, columns are data vectors
  % w is a M x 1 vector.
  % b is a scalar.
  % p = exp(w'*X)./(1+ exp(w'*X)) a 1 x N vector.
    
  cnst = log(1e200);
  tmp = w(:)' * X;
  p = zeros(1, size(X,2));
  idx1 = (tmp>cnst);
  % sidx1=sum(idx1)
  if (any(idx1))
    p(idx1) = 1;
  end
  idx0 = (tmp < -cnst);
  % sidx1=sum(idx0)
  if (any(idx0))
    p(idx0) = 0;
  end
  idx = (~idx0 & ~idx1);
  if (any(idx))
    p(idx) = exp(tmp(:, idx) + b);
    p(idx) = p(idx) ./ (1 + p(idx));
  end
  
  return
