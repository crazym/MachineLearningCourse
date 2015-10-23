function [ll, dll_dw, dll_db] = logisticNLP(x1, x2, w, b, alpha)
% [ll, dll_dw, dll_db] = logisticNLP(x1, x2, w, b, alpha)
% 
% Inputs:
%   x1 - array of exemplar measurement vectors for class 1.
%   x2 - array of exemplar measurement vectors for class 2.
%   w - an array of weights for the logistic regression model.
%   b - the bias parameter for the logistic regression model.
%   alpha - weight decay parameter
% Outputs:
%   ll - negative log probability (likelihood) for the data 
%        conditioned on the model (ie w).
%   dll_dw - gradient of negative log data likelihood wrt w
%   dll_db - gradient of negative log data likelihood wrt b


% YOUR CODE GOES HERE.

    sigmoid1 = logistic(x1, w, b);
    sigmoid2 = logistic(x2, w, b);

    n1 = size(x1, 2);
    n2 = size(x2, 2);

    x = [x1 x2];
    y = [ones(1, n1) zeros(1, n2)];

    
    % since yi is either 0 or 1
    ll=(1/2*alpha) * (w)' * w - (sum(log(sigmoid1)) + sum(log(1 - sigmoid2)));
    
    % find w
    dll_dw = w/alpha - x * (y - [sigmoid1 sigmoid2])';
    
    % find b
    dll_db = - sum(y - [sigmoid1 sigmoid2]);

end

