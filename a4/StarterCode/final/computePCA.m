function [V, D] = computePCA(imgs, mean)

    training_N = size(imgs, 2);
    dimension = size(imgs, 1);
    K = zeros(dimension);
    for i=1:training_N
        difference = imgs(:, i) - mean;
        K = K + difference * difference';
    end
    K = K ./ training_N;
    % dies not work since k need to be < n
    % [V, D] = eigs(K, 500, 'lm');
    [V_orig, D_orig] = eig(K);

    % refered to:
    % http://www.mathworks.com/matlabcentral/fileexchange/18904-sort-eigenvectors---eigenvalues
    D=diag(sort(diag(D_orig),'descend')); % make diagonal matrix out of sorted diagonal values of input D
    [c, ind]=sort(diag(D_orig),'descend'); % store the indices of which columns the sorted eigenvalues come from
    V=V_orig(:,ind); % arrange the columns in this order

end

