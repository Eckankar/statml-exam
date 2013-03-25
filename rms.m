function [ r ] = rms(expected, actual)
    N = size(expected, 1);
    r = 0;
    for i = 1:N
        r = r + (expected(i) - actual(i))^2;
    end
    r = sqrt(r / N);
end