function overview = dataset_overview(Market, Market_Title)
% This function computes the mean, median, standard deviation, and quantiles for the strikes, call prices, and put prices
% of a given dataset. The function returns a structure with the computed values.
%
% INPUT
%
% Market: Struct with Market data
% Market_Title: Title of the Market data
%
% OUTPUT
%
% overview: Struct with the computed values
%           - strikes: Mean, median, standard deviation, and quantiles for strikes
%           - call_prices: Mean, median, standard deviation, and quantiles for call prices
%           - put_prices: Mean, median, standard deviation, and quantiles for put prices

% Compute mean, median, std, and quantiles for strikes
strikes = [Market.strikes.value];
overview.strikes.mean = mean(strikes);
overview.strikes.median = median(strikes);
overview.strikes.std = std(strikes);
overview.strikes.quantile_05 = quantile(strikes, 0.05);
overview.strikes.quantile_95 = quantile(strikes, 0.95);

% Compute mean, median, std, and quantiles for call prices
call_prices = Market.midCall.value;
overview.call_prices.mean = mean(call_prices);
overview.call_prices.median = median(call_prices);
overview.call_prices.std = std(call_prices);
overview.call_prices.quantile_05 = quantile(call_prices, 0.05);
overview.call_prices.quantile_95 = quantile(call_prices, 0.95);

% Compute mean, median, std, and quantiles for put prices
put_prices = Market.midPut.value;
overview.put_prices.mean = mean(put_prices);
overview.put_prices.median = median(put_prices);
overview.put_prices.std = std(put_prices);
overview.put_prices.quantile_05 = quantile(put_prices, 0.05);
overview.put_prices.quantile_95 = quantile(put_prices, 0.95);

% Create a structure array for the results
results = struct();
results.Metric = {'Mean'; 'Median'; 'Standard Deviation'; 'Quantile 0.05'; 'Quantile 0.95'};
results.Strikes = [overview.strikes.mean; overview.strikes.median; overview.strikes.std; overview.strikes.quantile_05; overview.strikes.quantile_95];
results.CallPrices = [overview.call_prices.mean; overview.call_prices.median; overview.call_prices.std; overview.call_prices.quantile_05; overview.call_prices.quantile_95];
results.PutPrices = [overview.put_prices.mean; overview.put_prices.median; overview.put_prices.std; overview.put_prices.quantile_05; overview.put_prices.quantile_95];

% Convert the structure to a table
resultsTable = struct2table(results);

% Display the table
disp(['Dataset Overview for ' Market_Title]);
disp(resultsTable);

end