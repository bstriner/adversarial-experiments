function [ grid ] = GraphPoints( )
%GRAPHPOINTS Summary of this function goes here
%   Detailed explanation goes here
    xs = repmat(reshape(-2:0.1:2, [1,41]),41,1);
    ys = repmat(reshape(-2:0.1:2, [41,1]),1,41);
    xs = reshape(xs, [41*41 1]);
    ys = reshape(ys, [41*41 1]);
    grid = [xs ys];
end

