function [beta, converged] = logisticReg(X, y, v, winit)
  % Two class logistic regression
  % Input: X, data matrix  p+1 x N, 1st row of X is ones
  %        y, class indicator, 1 or 0, a 1 by N vector
  %        v, variance for a Gaussian prior on the weights
  %        winit, initial guess for weights (not required)
  % Output: beta = (p+1) vector of weights for logistic regression, where
  %        p(y=1 | x) = exp(beta' * x)/(1 + exp(beta' * x))
  %        p(y=0 | x) = 1 - p(y=1 | x)
  %
  % Allow for an isotrpoic prior on beta, v is a scalar variance
  % WHEN v == 0, then use NO prior.
  
  D = size(X,1);
  
  if ~exist('v','var')
      v = -1;
  elseif (v<0)
      disp(' problem: variance is not > 0 ');
  end
  
  % with a prior, it is often good to not place prior on bias offset
  WeightPriorOnly = 1;
  
  if ~exist('m','var')
      m = zeros(D,1);
  elseif WeightPriorOnly & (size(m,1) == D-1)
      m = [0;m];
  end
  
  if ~exist('winit','var')
      beta = zeros(D, 1);
  else
      beta = winit;
  end
  
  its = 0;
  converged = false;
  
  while (its < 500)
    p = logistic(X, beta);
    W = p .* (1-p);
    
    G = X * (y - p)';
    H = (X .* repmat(W, size(X,1), 1)) * (X');
    if (v>0)
        if WeightPriorOnly
            G = G - [0;beta(2:end)-m(2:end)] / v;
            H = H + diag([0,ones(1,D-1)]) / v;
        else
            G = G - (beta-m) / v;
            H = H + eye(D) / v;
        end
    end
    
    betaStep = pinv(H) * G;
    
    if (max(abs(betaStep)) < 1.0e-10)
      converged = true;
      break;
    end
    
    beta = beta + betaStep;
    
    its = its+1;
    
  end
  
  return

