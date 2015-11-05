function y = evalPolynomial(x,w)
% Evaluate the polynomial defined by the given weight vector at the
% given values of x.  This function should work even if x is a vector.
y = 0;
for i = 1:size(w)
    % follow the definition f(x) = \sum_{i=0}^{K} w(i) x^i
    y = y + w(i)*(x.^(i-1));
end
