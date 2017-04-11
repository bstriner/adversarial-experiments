function [ ynew ] = SubsCell( y, from, to)
%SUBSCELL Summary of this function goes here
%   Detailed explanation goes here
    ynew = y;
    for i = 1:size(from, 1)
        ynew = subs(ynew, from{i}, to{i});
    end
end

