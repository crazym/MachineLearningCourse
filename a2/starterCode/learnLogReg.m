function  [w,b] = learnLogReg(x1,x2,alpha)
%
% Inputs
%   x1  is an array of exemplars from class 1
%   x2  is an array of exemplars from class 2
%   alpha is an optional parameter for a weight decay regularizer
%
% Output
%   w  is a vector of estimated parameters for the logistic regressor
%   b  is the estimated bias term for the logistic regressor


%% TODO 1: 
%  Initialize the weights and bias.  The weights w are a vector whose dimension 
%  will depend on the dimension of each training exempla and the bias b is a scalar

w = ... 
b = ...

% Set this to one in order to (roughly) check whether your gradient is correct.
% Set to zero if you're confident that they're correct and you want 
% to skip the check
checkGradient = 1;

%%%% TODO 2: you need to implement the function, logisticNLP, that takes
%%%% data x1 and x2 from classes 1 and 2, and the weight vector w, 
%%%% and returns the negative log likelihood and the gradient of the
%%%% negative log likelihood.
%%%% Most of the code below can remain unchanged, up to the end of the inner
%%%% loop, but you should look at the code to make sure you understand it.

% Initial negative log-likelihood and gradient
[ll,dll_dw,dll_db] = logisticNLP(x1,x2,w,alpha);

% Use finite differences to compare numerical and analytical gradients.
if checkGradient
    fd_dll_dw = zeros(size(w));
    h = .001;
    for i=1:length(w)
        w2 = w;
        w2(i)=w2(i)+h;
        fd_dll_dw(i) = (logisticNLP(x1,x2,w2,b,alpha) - ll)/h;
    end
    fd_dll_db = (logisticNLP(x1,x2,w,b+h,alpha) - ll)/h
	
    disp('Analytic gradient:')
    disp(dll_dw)
    disp(dll_db)
    disp('Numerical gradient:')
    disp(fd_dll_dw)
    disp(fd_dll_db)
    disp('These should be nearly identical (i.e., within 1%)')
    pause
end

% Initial step-size
lambda = .001;

% Gradient descent inner-loop
l2 = [];
for t=1:1000
    l2 = [l2;ll];  %keep track of negative log-likelihoods in case you want to plot them
    lambda = lambda * 2;
    e = logisticNLP(x1,x2,w-lambda*dll_dw,b-lambda*dll_db,alpha);
    
    if sqrt(norm(dll_dw)^2 + norm(dll_db)^2) < eps
        warning('gradient is zero');
        break;
    end
    if isnan(e) || isinf(e)
        error('nan/inf error');
    end
     
    while (e >= ll) && (lambda > 0)
        lambda = lambda / 2;
        e = logisticNLP(x1,x2,w-lambda*dll_dw,b-lambda*dll_db,alpha);
    end
    if (lambda <= eps)
		warning('Infinitesimal lambda')
        fprintf(2,'lambda = %f\n',lambda);
        break;
    end
    
    w = w - lambda*dll_dw;
    b = b - lambda*dll_db;
    [ll,dll_dw,dll_db] = logisticNLP(x1,x2,w,b,alpha);
    
    fprintf(2,'step = %d, lambda = %f, ll=%f\n\r',t,lambda,ll);
end

