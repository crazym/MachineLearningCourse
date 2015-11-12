function p = logistic(X, w)
  % Evaluate logistic function
  % X is a M x N matrix, columns are data vectors
  % w is a M x 1 vector.
  % p = exp(w'*X)./(1+ exp(w'*X)) a 1 x N vector.
    
  cnst = log(1e200);
  if length(w) == size(X,1)
    tmp = w(:)' * X;
  elseif length(w) == size(X,1)+1 
    tmp = w(2:end)' * X + w(1);
  end
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
    p(idx) = exp(w' * X(:, idx));
    p(idx) = p(idx) ./ (1 + p(idx));
  end
  
  return
