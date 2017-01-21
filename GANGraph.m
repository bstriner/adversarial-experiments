function [  ] = GANGraph( path, epoch, x_real, g, d_W, d_b )
%GANGRAPH Summary of this function goes here
%   Detailed explanation goes here
    clf;
    hold on;
    ylim([-2 2]);
    xlim([-2 2]);
    grid = GraphPoints();
    values = sigmoid(mtimes(grid, d_W)+d_b);
    values = reshape(values, 41, 41);
    colormap('hot');
    image([-2 2], [-2 2], values, 'CDataMapping','scaled');
    colorbar;
    caxis([0 1]);
    % Plot real point
    scatter(x_real(1), x_real(2), 100, 'green', 'filled');
    % Plot generator point
    scatter(g(1), g(2), 100, 'blue', 'filled');
    %xticks([1 2 3]);
    %xticklabels({'Rock','Paper','Scissors'});
    %set(gca,'XTick',[1;2;3])
    %set(gca,'XTickLabel',{'Rock','Paper','Scissors'});
    %legend({'Generator', 'Target'});
    if ~exist(path,'dir')
        mkdir(path);
    end
    print(sprintf('%s/epoch-%04i.png',path,epoch),'-dpng');

end

