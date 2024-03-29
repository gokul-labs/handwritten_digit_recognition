jpg_files = fullfile("../images/", '*.*g');
images = dir(jpg_files);
data = [];

cd ../matlab/

layers = get_lenet();
load lenet.mat

for i = 1: length(images)
    img = imread(string(images(i).folder) + "/" + string(images(i).name));
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = 255 - img;
%     threshold_level = adaptthresh(img);
    img = imbinarize(img);
    [L, no_of_components] = bwlabel(img, 8);
%     figure;
%     vislabels(L);
    conn_comp = bwconncomp(img);
    boundaries = regionprops(conn_comp);
    for c = 1: no_of_components
        cropped_image = imcrop(img,boundaries(c).BoundingBox);
        cropped_image = imresize(cropped_image, [28, 28]);
%         imshow(cropped_image);
        cropped_image = cropped_image';
        data = [data, cropped_image(:)];
    end
end


predicted_labels = [];
expected_labels = [1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,7,1,7,3,5,5,8,1,10,4,2,7,8,3,7,...
    2,4,10,7,5,2,5,3,1,1,6,5,5,8,4,2,1,3,6,6,2,8,8,5,10,2,8,5,3,10,2,6,4,5,1,3,6,10,5,5,2,2];
prob_prediction = [];

data = reshape(data, 28,28,1,[]);
layers{1}.batch_size = size(data, 4);
[output, P] = convnet_forward(params, layers, data);

cd ../ec/

P = reshape(P, [], layers{1}.batch_size);
for b = 1:layers{1}.batch_size
    predictions_for_one_sample = P(:,b);
    [max_value, index_of_max_value] = max(predictions_for_one_sample(:));
    predicted_labels(b) = index_of_max_value;
    prediction_probs = predictions_for_one_sample(:);
    prob_prediction(b) = prediction_probs(index_of_max_value);
end

disp(expected_labels);
disp(predicted_labels);
disp(prob_prediction);

% Uncomment to visualize the predictions
% C = confusionmat(expected_labels(:),predicted_labels(:));
% disp(C);
% disp(confusionchart(C));
