function prob = posterior(Xmax, M, N )

prob = zeros(1, M);
denominator_summation = 0;
for i = Xmax:M
    denominator_summation = denominator_summation + (1/i)^N;
end

for L = Xmax:M
    prob(:, L) = (1/L)^N / denominator_summation;

end

