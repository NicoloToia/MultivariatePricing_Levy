function [discounts]=intExtDF(discounts, dates, targetDates)
% intExtDF: Interpolates (linear) or extrapolates (flat) the zero rates curve for a given date,
%           if the discount is present for the given date it returns the
%           value without interpolating or extrapolating
%
% INPUT
% discounts  : discount factors curve for preceding values (Bootstrap) 
% dates      : dates of discount factors curve bootstrap
% targetDate : corrisponding date of the discount requested
%
% OUTPUT
% discount   : discont factor found by matching/interpolating/extrapolating
%              the zero rates curve


% Zero rates are interpolated with yearfrac Convention ACT/365
ACT_365 = 3;

% compute the zero rates
zeroRates = -log(discounts)./yearfrac(dates, dates(1), ACT_365);
% interpolate
searchRates = interp1(dates, zeroRates, targetDates, 'linear', zeroRates(end));
% reconstruct the discount factors
discounts = exp(-searchRates.*yearfrac(targetDates, dates(1), ACT_365));

end