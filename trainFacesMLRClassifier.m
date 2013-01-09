function[wSoftmax] = trainFacesMLRClassifier(training_data,training_labels,unlabeled_images)
% Trains a multinomial logistic regression classifier to predict the labels
% for data_test. Note that this is purely supervised and does not use
% unlabeled data in any way.
[num_rows,num_cols,num_train] = size(training_data);

% Reshape the data to the standard num_data x num_dimensions format
X = reshape(training_data,num_train,num_rows*num_cols);

% Add bias
X = [ones(num_train,1), X];

% Convert targets from 1 of K to vector representation
[junk,y] = max(training_labels,[],2);
nClasses = size(training_labels,2);
nVars = num_rows*num_cols;

% Create a pointer to the classifier function
funObj = @(W)SoftmaxLoss2(W,X,y,nClasses);
lambda = 1e-4*ones(nVars+1,nClasses-1);
lambda(1,:) = 0; % Don't penalize biases
options = [];
fprintf('Training multinomial logistic regression model...\n');
wSoftmax = minFunc(@penalizedL2,zeros((nVars+1)*(nClasses-1),1),options,funObj,lambda(:));
wSoftmax = reshape(wSoftmax,[nVars+1 nClasses-1]);
wSoftmax = [wSoftmax zeros(nVars+1,1)];


% This version of softmax regression only requires (nVars+1)x(nClasses-1)
% parameters as opposed to (nVars+1)x(nClasses)
wInit = zeros((nVars+1)*(nClasses-1),1);
wSoftmax = minFunc(funObj,wInit,options);