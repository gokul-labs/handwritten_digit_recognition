function [param_grad, input_od] = inner_product_backward(output, input, layer, param)
input_od = zeros(size(input.data));
batch_size = input.batch_size;
% Replace the following lines with your implementation.
param_grad.b = zeros(size(param.b));
param_grad.w = zeros(size(param.w));

gw = input.data * output.diff';
gb = sum(output.diff, 2);
param_grad.w = gw;
param_grad.b = gb';
input_od = param.w * output.diff;

end
