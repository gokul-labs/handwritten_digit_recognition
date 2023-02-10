function [output] = conv_layer_forward(input, layer, param)
% Conv layer forward
% input: struct with input data
% layer: convolution layer struct
% param: weights for the convolution layer

% output:

h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;
num = layer.num;
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;

assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')
input_n.height = h_in;
input_n.width = w_in;
input_n.channel = c;

%% Fill in the code
% Iterate over the each image in the batch, compute response,
% Fill in the output datastructure with data, and the shape.

%  result = zeros([h_out,w_out,num,batch_size]);
%  input.data = reshape(input.data, h_in, w_in ,c, batch_size);
%  for i = 1:batch_size
%      input_n.data = reshape(input.data(:,:,:,i), h_in, w_in, c);
%      input_n.data = padarray(input_n.data,[pad, pad], 0);
%      input_n.data = dlarray(input_n.data, "SSC");
%      weights = reshape(param.w, k,k,c,num);
%      bias = param.b';
%      Y = dlconv(input_n.data, weights,bias);
%      output_n.data = extractdata(Y);
%      output_n.data = reshape(output_n.data, h_out, w_out, num);
%      result(:,:,:,i) = output_n.data;
%  end

result = zeros([h_out,w_out,num,batch_size]);
input.data = reshape(input.data, h_in, w_in ,c, batch_size);
for i = 1:batch_size
    input_n.data = reshape(input.data(:,:,:,i), h_in, w_in, c);
    col = im2col_conv(input_n, layer, h_out, w_out);
    col = reshape(col, k*k*c, h_out*w_out);
    weights = reshape(param.w, k*k*c,num);
    for f = 1:num
        weight_temp = weights(:,f);
        temp_result = col' * weight_temp + param.b(f);
        result(:,:,f,i) = reshape(temp_result, h_out, w_out);
    end
end
output.data = result;
output.height = h_out;
output.width = w_out;
output.channel = num;
output.batch_size = batch_size;
output.data = reshape(output.data, [], batch_size);
end

