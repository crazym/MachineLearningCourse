%%
p1 = posterior(700, 999, 1 );
p2 = posterior(700, 999, 10 );
p3 = posterior(700, 999, 100 );

figure;hold on;
plot(1:999,p1,'-r');
plot(1:999,p2,'-g');
plot(1:999,p3,'-b');
xlabel('L - largest ID number');
ylabel('P(L|X_{1:N}) - posterior probability');  
title('Q4 - Posterior distribution of P(L|X_{1:N}) regarding different N');
legend('N = 1','N = 10','N = 100');

%

b1 = bayes(700, 999, 1 );
b2 = bayes(700, 999, 10 );
b3 = bayes(700, 999, 100 );

%%

pred = prediction(700, 999, 10);
figure;hold on;
plot(1:999,[1/700*ones(1, 700) zeros(1, 999-700)],'r');
plot(1:999,[1/b2*ones(1, floor(b2)) zeros(1, 999-floor(b2))],'b');
plot(1:999,pred,'g');
xlabel('ID number of the next person');
ylabel('probability');
title('Probability distribution of new ID number');
legend('condition on L_{MAP}','condition on L_{Bayes}','condition on X_{1:N}');

