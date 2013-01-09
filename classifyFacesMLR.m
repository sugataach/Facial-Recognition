function[test_set] = classifyFacesMLR(wSoftmax,test_data,nClasses)
% Evaluates an MLR classifier for the Toronto Faces Dataset on test data
% Reshape parameters from a vector to a matrix, the last column should
% contain zeros in this version (don't worry about why, it's basically
% removing redundancy in the original softmax formulation)
[num_rows,num_cols,num_test] = size(test_data);
nVars = num_rows*num_cols;
wSoftmax = reshape(wSoftmax,[nVars+1 nClasses-1]);
wSoftmax = [wSoftmax zeros(nVars+1,1)];

% Make predictions on the test set
X_test = reshape(test_data,num_test,num_rows*num_cols);
X_test = [ones(num_test,1), X_test];

[junk test_set] = max(X_test*wSoftmax,[],2);