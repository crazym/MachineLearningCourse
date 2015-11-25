% Simple Eigen-eyes detector. Load data (thanks to Francisco Estada  
% and Allan Jepson for allowing us to use this dataset).

load trainSet
load testSet

% the variables defined in the .mat files are:
% eyeIm - a 500 x n array, each COLUMN contains a vector that
%         represents an eye image
% nonIm - a 500 x m array, each COLUMN contains a vector that
%	  represents a non-eye image
% sizeIm - size of the eye and non eye images [y x]
who

% Normalize brightness to [0 1]
eyeIm=eyeIm/255;
nonIm=nonIm/255;
testEyeIm=testEyeIm/255;
testNonIm=testNonIm/255;

% You can display images from eyeIm or nonIm using;
%
% imagesc(reshape(eyeIm(:,2),sizeIm));axis image;colormap(gray)
%  - where of course you would select any column

% We will first see how far we can get with classification
% on the original data using kNN. The task is to distinguish
% eyes from non-eyes. This is useful to gain insight about
% how hard this problem is, and how much we can improve
% or lose by doing dimensionality reduction.

% Generate training and testing sets with classes for kNN,
% we need eye images to be on ROWS, not COLUMNS, and we also 
% need a vector with class labels for each

trainSet=[eyeIm'
          nonIm'];
trainClass=[zeros(size(eyeIm,2),1)
            ones(size(nonIm,2),1)];

testSet=[testEyeIm'
         testNonIm'];
testClass=[zeros(size(testEyeIm,2),1)
            ones(size(testNonIm,2),1)];

% Compute matrix of pairwise distances (this takes a while...)
d=som_eucdist2(testSet,trainSet);
% 
% Compute kNN results, I simply chose a reasonable value
% for K but feel free to change it and play with it...
K=5;
[C,P]=knn(d,trainClass,K);

% Compute the class from C (we have 0s and 1s so it is easy)
class=sum(C,2);	  		% Add how many 1s there are
class= (class>(K/2));   % Set to 1 if there are more than K/2
				        % ones. Otherwise it's zero

% Compute classification accuracy: We're interested in 2 numbers:
% Correct classification rate - how many eyes were classified as eyes
% False-positive rate: how many non-eyes were classified as eyes

fprintf(2,'Correct classification rate:\n');
correctEye_knn=length(find(class(1:size(testEyeIm,2))==0))/size(testEyeIm,2)

fprintf(2,'False positive rate:\n');
falseEye_knn=length(find(class(size(testEyeIm,2)+1:end)==0))/size(testNonIm,2)

% Keep in mind the above figures! (and the kNN process, you'll
% have to do it again on the dimension-reduced data later on.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% PCA PART: Your task begins here!
% Do PCA on eyes and non-eyes to generate models for recognition


%%% TO DO:
% First, compute the mean over the eye and non-eye images
% i.e. compute the mean eye image, and the mean non-eye image
eyeMean = mean(eyeIm, 2);
noneyeMean = mean(nonIm, 2);
%%% TO PRINT:
% Plot the mean eye and mean non-eye images and hand the
% printouts in with your report.

%%% TO DO:
% Now, perform the PCA computation as discussed in lecture over the 
% training set. You will do this separately for eye images and non-eye 
% images. This will produce a set of eigenvectors that represent eye 
% images, and a different set of eigenvectors for non-eye images.

[eyeVec, eyeDiag] = computePCA(eyeIm, eyeMean);
[noneyeVec, noneyeDiag] = computePCA(nonIm, noneyeMean);

%%% TO PRINT:
% Display and print out the first 5 eigenvectors for eyes and non-eyes 
% (i.e. the eigenvectors with LARGEST 5 eigenvalues, make sure you sort 
% the eigenvectors by eigenvalues!)
disp('first 5 eigenvectors for eyes')
eyeVec(:, 1:5);
disp('first 5 eigenvectors for non-eyes')
noneyeVec(:, 1:5);


%%% TO DO:
% Now you have two PCA models: one for eyes, and one for non-eyes. 
% Next we will project our TEST data onto our eigenmodels to obtain 
% a low-dimensional representation first we choose a number of PCA
% basis vectors (eigenvectors) to use:

PCAcomp=10;	% Choose 10 to start with, but you will 
		% experiment with different values of 
		% this parameter and see how things work

% To compute the low-dimensional representation for a given
% entry in the test test, we must do 2 things. First, we subtract
% the mean, and then we project that vector on the transpose of 
% the PCA basis vectors.  For example, say you have an eye image
%
% vEye=testSet(1,:);   % This is a 1x500 row vector
%
% The projections onto the PCA eigenvectors are:
%
% coeffEye=eyeVec(:,1:PCAcomp)'*(vEye'-eyeMean);
% coeffNonEye=noneyeVec(:,1:PCAcomp)'*(vNonEye'-noneyeMean);
%
% You need to compute coefficients for BOTH the eye and non-eye 
% models for each testSet entry, i.e. for each testSet image you 
% will end up with (2*PCAcomp) coefficients which are the projection 
% of that test image onto the chosen eigenvectors for eyes and non-eyes.

% Since we are going to use the KNN classifier demonstrated above, 
% you might want to place all the of the test coefficients into one 
% matrix.  You would then end up with a matrix that has one ROW for 
% each image in the testSet, and (2*PCAcomp) COLUMNS, one for each 
% of the coefficients we computed above.
test_N = size(testSet, 1);
coeffsEye = zeros(test_N, PCAcomp);
coeffsNonEye = zeros(test_N, PCAcomp);
for i=1:test_N
    vEye=testSet(i,:);
    coeffsEye(i, :)=eyeVec(:,1:PCAcomp)'*(vEye'-eyeMean);
    coeffsNonEye(i, :)=noneyeVec(:,1:PCAcomp)'*(vEye'-noneyeMean);
end

coeffs = [coeffsEye coeffsNonEye];

%%% TO DO:
% Then do the same for the training data.  That is, compute the 
% PCA coefficients for each training image using both of the models.
% Then you will have low-dimensional test data and training data
% ready for the application of KNN, just as we had in the KNN example
% at the beginning of this script.
train_N = size(trainSet, 1);
coeffsEye_train = zeros(train_N, PCAcomp);
coeffsNonEye_train = zeros(train_N, PCAcomp);
for i=1:train_N
    vEye=trainSet(i,:);
    coeffsEye_train(i, :)=eyeVec(:,1:PCAcomp)'*(vEye'-eyeMean);
    coeffsNonEye_train(i, :)=noneyeVec(:,1:PCAcomp)'*(vEye'-noneyeMean);
end
coeffs_train = [coeffsEye_train coeffsNonEye_train];
%%% TO DO
% KNN classification: 
% Repeat the procedure at the beginning of this script, except
% instead of using the original testSet data, use the 
% coefficients for the training and testing data, and the same
% class labels for the training data that we had before
%

% Compute classification accuracy: We're interested in 2 numbers:
% Correct classification rate - how many eyes were classified as eyes
% False-positive rate: how many non-eyes were classified as eyes

%??????????????? input ?????????
% [correctEye_lknn, falseEye_lknn] = knnRates(eyeIm, nonIm, testEyeIm, testNonIm);
% 
% fprintf(2,'Correct classification rate:\n');
% disp(correctEye_lknn)
% fprintf(2,'False positive rate:\n');
% disp(falseEye_lknn)

% Compute matrix of pairwise distances (this takes a while...)
dl=som_eucdist2(coeffs,coeffs_train);

[Cl,Pl]=knn(dl,trainClass,K);

% Compute the class from C (we have 0s and 1s so it is easy)
classLowDim=sum(Cl,2);	  		% Add how many 1s there are
classLowDim= (classLowDim>(K/2));   % Set to 1 if there are more than K/2
                        % ones. Otherwise it's zero

% Compute classification accuracy: We're interested in 2 numbers:
% Correct classification rate - how many eyes were classified as eyes
% False-positive rate: how many non-eyes were classified as eyes
fprintf(2,'Correct classification rate for Low-dimensional data:\n');
correctEye_knnl=length(find(classLowDim(1:size(testEyeIm,2))==0))/size(testEyeIm,2)

fprintf(2,'False positive rate for Low-dimensional data:\n');
falseEye_knnl=length(find(classLowDim(size(testEyeIm,2)+1:end)==0))/size(testNonIm,2)


%%% TO PRINT:
% Print the classification accuracy and false-positive rates for the
% kNN classification on low-dimensional data and compare with the
% results on high-dimensional data.
%
% Discuss in your report: 
% - Are the results better? worse? is this what you expected?
% - why do you think the results are like this?
%

%%% TO DO:
% Finally, we will do classification directly from the PCA models
% for eyes and non-eyes.
%
% The idea is simple: Reconstruct each entry in the testSet
% using the PCA model for eyes, and separately the PCA model
% for non-eyes. Compute the squared error between the original
% entry and the reconstructed versions, and select the class
% for which the reconstruction error is smaller. It is assumed
% that the PCA model for eyes will do a better job of
% reconstructing eyes and the PCA model for non-eyes will
% do a better job for non-eyes (but keep this in mind:
% there's much more stuff out there that is not an eye
% than there are eyes!)
%
% To do the reconstruction, let's look at a vector from the
% coefficients we computed earlier for the training set;
%
% Reconstruction
%
% vRecon_eye= eyeMean + sum_k (eye_coeff_k * eye_PCA_vector_k);
%
% i.e. the mean eye image, plus the sum of each PCA component 
% multiplied by the corresponding coefficient. One can also replace
% the sum with a matrix-vector product.  Note: If you don't add 
% the mean image component back this won't work!
%
% Likewise, for the reconstruction using the non-eye model
%
% vRecon_noneye= nonMean + sum_k (noneye_coeff_k * noneye_PCA_vector_k)
%

%%% TO DO:
%
% Compute the reconstruction for each entry using the PCA model for eyes
% and separately for non-eyes, compute the error between these 2 
% reconstructions and the original testSet entry, and select the class
% that yields the smallest error.

classPCA = zeros(test_N, 1);
for i=1:test_N
    % vRecon_eye= eyeMean + sum_k (eye_coeff_k * eye_PCA_vector_k);
    vRecon_eye= eyeMean + eyeVec(:, 1:PCAcomp) * coeffsEye(i,:)';
    % vRecon_noneye= noneyeMean + sum_k (noneye_coeff_k * noneye_PCA_vector_k);
    vRecon_noneye= noneyeMean + noneyeVec(:, 1:PCAcomp) * coeffsNonEye(i,:)';
    % compute the square error between two reconstructions and the original
    % testSet entry
    dist_eye = norm(vRecon_eye - testSet(i,:)');
    dist_noneye = norm(vRecon_noneye - testSet(i,:)');
    % if noneye error is smaller, set the class to 1, o.w. 0
    classPCA(i) = dist_noneye < dist_eye;
end

fprintf(2,'PCA correct classification rate for PCAcomp = %d :\n', PCAcomp);
correctEye_pca=length(find(classPCA(1:size(testEyeIm,2))==0))/size(testEyeIm,2)

fprintf(2,'PCA False positive rate for PCAcomp = %d :\n', PCAcomp);
falseEye_pca=length(find(classPCA(size(testEyeIm,2)+1:end)==0))/size(testNonIm,2)

%%% TO PRINT:
%
% Print the correct classification rate and false positive rate for
% the PCA based classifier and the low-dimensional kNN classifier
% using PCAcomps=5,10,15,25, and 50
%
% Plot a graph of the kNN classification rate for the low-dimensional
% KNN classifier VS the number of PCA components (for the 5 values of 
% PCAcomps requested). 
%
% Discuss in your Report:
% - Is there a value for PCAcomps (or set of values) for which low-dimensional
%   kNN is better than full dimensional kNN? 
% - why do you think that is?
%
% Plot graphs of correct classification rate and the false-positive rate 
% fr the PCA-reconstruction classifier vs the number of PCA components.
%
% Discuss in your Report:
% - Which classifier gives the overall best performance?
% - What conclusions can you draw about the usefulness of dimensionality
%   reduction?
% - Which classifier would you use on a large training set
%   consisting of high-dimensional data?
% - Which classifier would you use on a large training set
%   of low-dimensional (e.g. 3-D) data?
% - why?
% - Summarize the advantages/disadvantages of each classifier!
%
