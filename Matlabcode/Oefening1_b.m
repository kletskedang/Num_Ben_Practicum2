load('DatasetCV.mat');

max_n = 5;

A = zeros(2*max_n, size(x, 1));
for i = 1:max_n
    A(2*i-1, :) = x'.^i; 
    A(2*i, :) = y'.^i;
end

B = cat;


