layers = get_lenet();
load lenet.mat
% load data
% Change the following value to true to load the entire dataset.
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;
 
 
layers{1}.batch_size = 1;
img = xtest(:, 1);
img = reshape(img, 28, 28);
figure;
imshow(img')
 
%[cp, ~, output] = conv_net_output(params, layers, xtest(:, 1), ytest(:, 1));
output = convnet_forward(params, layers, xtest(:, 1));
output_conv = reshape(output{2}.data, output{2}.height, output{2}.width, output{2}.channel);
% Fill in your code here to plot the features.
figure;
set(gcf,'position',[0,0,600,600]);
for c = 1:output{2}.channel
    subplot(4,5,c)
    img_temp = output_conv(:,:,c);
    imshow(img_temp')
end

figure;
output_relu = reshape(output{3}.data, output{3}.height, output{3}.width, output{3}.channel);
set(gcf,'position',[0,0,600,600]);
for c = 1:output{3}.channel
    subplot(4,5,c)
    img_temp = output_relu(:,:,c);
    imshow(img_temp')
end


