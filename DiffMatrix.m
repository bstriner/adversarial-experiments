function [ d ] = DiffMatrix( y, x )
%DIFFMATRIX Summary of this function goes here
%   Detailed explanation goes here
    s=size(x);
    d = sym('d',s);
    for i = 1:s(1)
        for j = 1:s(2)
            d(i,j) = diff(y,x(i,j));
        end
    end
end

