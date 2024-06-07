function state = customGaplotbestf(options, state, flag)
    % Custom plot function for GA
    persistent bestMeanStdPlot;
    
    switch flag
        case 'init'
            % Initialize the plot
            bestMeanStdPlot = figure;
            hold on;
            grid on;
            xlabel('Generation');
            ylabel('Fitness Value');
            title('Best, Mean, and Std of Fitness Value per Generation');
            legend({'Best', 'Mean', 'Std'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
        case 'iter'
            % Update the plot with current generation data
            figure(bestMeanStdPlot);
            best = min(state.Score);
            meanVal = mean(state.Score);
            stdVal = std(state.Score);
            plot(state.Generation, best, 'r.-');
            plot(state.Generation, meanVal, 'b.-');
            plot(state.Generation, stdVal, 'k.-');
        case 'done'
            % Finalize the plot
            hold off;
    end
end
