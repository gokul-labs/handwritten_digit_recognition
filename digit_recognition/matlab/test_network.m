%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network
% Modify the code to get the confusion matrix
predicted_labels = [];
expected_labels = [ytest(:)];

%Prediction information for all batches
for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    P = reshape(P, [], layers{1}.batch_size);
    for b = 1:layers{1}.batch_size
        predictions_for_one_sample = P(:,b);
        [max_value, index_of_max_value] = max(predictions_for_one_sample(:));
        predicted_labels(i+b-1) = index_of_max_value;
    end
end

C = confusionmat(expected_labels(:),predicted_labels(:));
disp(C);
disp(confusionchart(C));
