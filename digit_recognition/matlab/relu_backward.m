function [input_od] = relu_backward(output, input, layer)
% Replace the following line with your implementation.
input.data = arrayfun(@(x) x>0, input.data);
input_od = input.data .* output.diff;
output.batch_size = input.batch_size;
output.channel = input.channel;
output.width = input.width;
output.height = input.height;
end
