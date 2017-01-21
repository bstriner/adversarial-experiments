function [ y ] = sigmoid( x )
%SIGMOID Sigmoid function

    y = 1.0 ./ (1.0 + exp(-x));

end

