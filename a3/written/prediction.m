function pred = prediction(Xmax, M, N)
numerator = 0;
constant = 0;
prob = [];
for k = Xmax:M
    constant = constant + (1/k).^N;
end

for i = 1:M
    if i <= Xmax
        for k = Xmax:M
            numerator = numerator + (1/k).^(N + 1);            
        end
    
    else
        for k = i:M
            numerator = numerator + (1/k).^(N + 1);
        end
    end
    prob = [prob numerator/constant];
    numerator = 0;     
pred = prob;
end
