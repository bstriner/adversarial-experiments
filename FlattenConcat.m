function [ y ] = FlattenConcat( x )
%FLATTENCONCAT Summary of this function goes here
%   Detailed explanation goes here
    h = Flatten(x{1});
    for i = 2:size(x,1)
        h = [h; Flatten(x{i})];
    end
    y = h;
end

