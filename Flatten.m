function [ y ] = Flatten( x )
%FLATTEN Summary of this function goes here
%   Detailed explanation goes here
    y = reshape(x, numel(x), 1);
end

