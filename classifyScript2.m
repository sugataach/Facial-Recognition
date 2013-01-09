% This script will train a softmax regression classifier on the Toronto
% Faces Dataset.

% Load the training set
load training;
load unlabeled_images;
% Convert the data to double otherwise softmax loss throws an error
training_data = double(tr_images);
training_labels = double(tr_labels);

[num_rows,num_cols,num_faces] = size(training_data);

% Randomly split the data into a training a validation set
%rand('state',1);
%randn('state',1);
%ind = randperm(num_faces);
%num_train = ceil(num_faces*(4/5));
%num_valid = num_faces - num_train;
% Partitioning validation data from training dataset
%validation_data = training_data(:,:,ind(num_train+1:num_faces));
% Partitioning validation labels from training labels
%validation_labels = training_labels(ind(num_train+1:num_faces),:);
% Creating new training set by shrinking the old training set
%training_data = training_data(:,:,ind(1:num_train));
% Creating new tr_labels_valid by removing tr_labels_valid from old tr_labels
%training_labels = training_labels(ind(1:num_train),:);

% Learn the parameters on the training set
wSoftmax = trainFacesMLRClassifier(training_data,training_labels,unlabeled_images);
% Save the learned model to a mat file
save trainedModel wSoftmax;

load val_images;
validation_data = double(val_images);
% Evaluate the learned parameters on the validation set and report the
% validation set accuracy
[test_set] = classifyFacesMLR(wSoftmax,validation_data,7);
%tr_labels_valid = oneOfK2Num(tr_labels_valid);
%validAccuracy = sum(sum(test_set == validation_labels))/size(validation_labels,1)

% Save the results to a mat file
save test.mat test_set