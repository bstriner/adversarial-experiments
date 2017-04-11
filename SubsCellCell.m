function [ ynew ] = SubsCellCell( ys, from, to )
%SUBSCELLCELL Summary of this function goes here
%   Detailed explanation goes here
    ynew = cell(size(ys));
    for i = 1:size(ys,1)
        ynew{i} = double(vpa(SubsCell(ys{i}, from, to)));
    end
end

