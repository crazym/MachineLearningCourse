function [correctEye_knn, falseEye_knn] = knnRates(eyeIm, nonIm, testEyeIm, testNonIm)

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

    correctEye_knn=length(find(class(1:size(testEyeIm,2))==0))/size(testEyeIm,2);
    falseEye_knn=length(find(class(size(testEyeIm,2)+1:end)==0))/size(testNonIm,2);

end