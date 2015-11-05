function m = bayes(Xmax, M, N)

m = 0;
denominator = 0;

for i = Xmax:M
    denominator = denominator + (1/i)^N;
end

for L = Xmax:M
    m = m + ((1/L)^(N - 1)) * (1/ denominator);
end

end

