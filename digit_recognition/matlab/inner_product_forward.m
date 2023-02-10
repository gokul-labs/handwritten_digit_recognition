function [output] = inner_product_forward(input, layer, param)
input.data = reshape(input.data, [], input.batch_size);
d = size(input.data, 1);
k = size(input.data, 2); % batch size
n = size(param.w, 2);

output.data = zeros([n, k]);
output.data = param.w' * input.data;

output.height = input.height;
output.width = input.width;
output.channel = input.channel;
output.batch_size = input.batch_size;

output.data = reshape(output.data, [], input.batch_size);
end
