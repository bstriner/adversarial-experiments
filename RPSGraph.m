function [ ] = RPSGraph(path, epoch, a, b )
%RPSGRAPH Graph for rock paper scissors
%   Create a bar chart and save it to PNG file
    clf;
    data = [a b];
    bar(data);
    ylim([0 1]);
    %xticks([1 2 3]);
    %xticklabels({'Rock','Paper','Scissors'});
    set(gca,'XTick',[1;2;3])
    set(gca,'XTickLabel',{'Rock','Paper','Scissors'});
    legend({'Player A', 'Player B'});
    if ~exist(path,'dir')
        mkdir(path);
    end
    print(sprintf('%s/epoch-%04i.png',path,epoch),'-dpng');
end

