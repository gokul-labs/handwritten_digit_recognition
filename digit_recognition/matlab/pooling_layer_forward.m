function [output] = pooling_layer_forward(input, layer)
h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;

h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;


output.height = h_out;
output.width = w_out;
output.channel = c;
output.batch_size = batch_size;

% Replace the following line with your implementation.
output.data = zeros([h_out, w_out, c, batch_size]);
input.data = reshape(input.data, h_in, w_in, c, batch_size);

input_n.height = h_in;
input_n.width = w_in;
input_n.channel = c;

for i = 1:batch_size
    input_n.data = input.data(:,:,:,i);
    col = im2col_conv(input_n, layer, h_out, w_out);
    col = reshape(col, k*k, c, h_out, w_out);
    for ch = 1:c
        for w=1:w_out
            for h=1:h_out
                temp_result = max(col(:,ch,h,w));
                output.data(h,w,ch,i) = temp_result;
            end
        end
    end
end
output.data = reshape(output.data, [], batch_size);
end

