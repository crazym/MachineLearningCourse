% This is a small script to help you review some of the
% concepts covered during class for the linear regression
% lecture
%
% FEG, 09-09-2008 
% DJF, 09-12-2009
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% First let's set up some data. In this example the data comes from
% a simple line with slope .75 and Y-axis intercept=2

w=.75;
b=2;

% Let's get a few locations randomly along this line between x=0 and x=5;
% 
% x=5*rand(15,1)   % Get 15 random x locations between 0 and 5
% 	
% y=(w*x)+b	  % Compute the exact value of y from the line equation

% Plot the noiseless data

% figure(1);clf;
% plot(x,y,'bo');hold on;
% title('Data sampled from a line. Blue is noiseless, magenta is with noise');
% 
% % Looks nice, but unfortunately we don't generally work with perfect
% % data, sources of error: Measurement problems, transmission distortion,
% % random noise or influences from the environment.
% 
% %% We will look at typical noise models later on, in particular, we will
% % use Gaussian functions to model nose quite often. Let's for now
% % use Matlab to generate some random Gaussian noise, with mean of
% % zero and a standard deviation of .1 
% 
% y_noisy=y+(.1*randn(length(y),1));
% 
% % Plot it on the same figure in red to see what it looks like
% 
% plot(x,y_noisy,'ms');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now, let's have some fun with Matlab:
%
% Go to the Tools menu on the figure menu, and select Basic Fitting
% This will open up a menu
% - Select the data source corresponding to the BLUE points first
%    (ie data set 1)
% - Then check on the 'linear' checkbox
%   You will see the best-fit line estimated from our samples
%   using the same procedure we covered in lecture
% - Try fitting the data with other functions
%   
% Question: Do all the functions approximate the data equally well?
%           what is going on here?
%           Is there a point at which Matlab complains about something?
% 10th poly: Polynomial is badly conditioned.
% Add points with distince X values, select a polynomial with a lower
% degree, or select "Center and scale data"
%          
% - Now, select the data source corresponding to the MAGENTA points
% - Try the linear fit again
% - With the linear fit on, try other functions
%
% Question: Are the results similar to the noiseless case? can all
%           functions approximate the data equally well?
%           if not, which function does a better job?
% No, approximation curves varies a lot


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear this plot and let's get some more data and more noise

clf;

% Now let's see what happens when we have more data, and also, more noise!
% Let's take twice the amount of data

x=5*randn(30,1);
y=(w*x)+b;
y_noisy_1=y+(.1*randn(length(y),1));	% Random noise stDev=.1
y_noisy_2=y+(.75*randn(length(y),1));	% Random noise stDev=.75

figure(1);
plot(x,y_noisy_1,'bs');
hold on;
plot(x,y_noisy_2,'m*');
title('Noisy data with different amounts of noise');

% Try out fitting these data with different functions
% like above... do you get similar results? overall,
% what would you say is the function that best 
% approximates the data?


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Least Squares Regression 
%
% We now know Matlab can solve this problem, but let's do it
% ourselves using the Least Squares closed-form solution for
% parameters w and b that we discussed in class
%
% Remember, the exact paramaters for out line are
% w=.75, b=2
%
% From our notes: w=[sum_i (y_i-Y)(x_i-X)]/sum_i (x_i-X)^2
%                 b=Y-wX
%
% where Y is the average y value for the input samples
% and X is the average x value for the input samples
%

% Let's look at the noiseless case first (y and x in our workspace)

X_noiseless=mean(x);	% Compute average X
Y_noiseless=mean(y);	% Compute average Y

w_noiseless=sum((y-Y_noiseless).*(x-X_noiseless))/sum((x-X_noiseless).^2)
b_noiseless=Y_noiseless-(w_noiseless*X_noiseless)

% We get the exact solution, which is as we expected since in the noiseless
% case the optimal solution indeed corresponds to the original model.

% Now see what happens with the noisy cases, first, with little noise
% (y_noisy1 in our workspace)

% The mean x value doesn't change! notice that we only added noise to
% the y measurements. Hence, we don't need to recompute the averade
% x value and we simply use X_noiseless.

Y_noisy1=mean(y_noisy_1);	% Mean y for noisy 1 data

w_noisy1=sum((y_noisy_1-Y_noisy1).*(x-X_noiseless))/sum((x-X_noiseless).^2)
b_noisy1=Y_noisy1-(w_noisy1*X_noiseless)

% Close! but the effect of noise is clear, see what happens with
% y_noisy_2

Y_noisy2=mean(y_noisy_2);	% Mean y for noisy 2 data

w_noisy2=sum((y_noisy_2-Y_noisy2).*(x-X_noiseless))/sum((x-X_noiseless).^2)
b_noisy2=Y_noisy2-(w_noisy2*X_noiseless)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Often the best way to try to formulate and solve problems in Matlab 
%  is in matrix vector form.  As explained in the notes, the estimator
%  for coefficients of the linear model can also be expressed in terms
%  of the pseudo-inverse.   That is, the normal equations derived from 
%  taking the gradient of the energy function take the form
%  Let's follow that approach (look how little code it requires):

X = [x ones(size(x))];

% here's the noiseless solution
w1 = X \ y

% here are the two noisy cases
w2 = X \ y_noisy_1
w3 = X \ y_noisy_2

% note that the results we get here are identical to those above.

% how much error is there when we run the model on the training data?
E = norm(y - X*w1)
E = norm(y_noisy_1 - X*w2)
E = norm(y_noisy_2 - X*w3)

% As expected, there is almost no error for the noiseless case, but 
% some significant error with the noisy data.



