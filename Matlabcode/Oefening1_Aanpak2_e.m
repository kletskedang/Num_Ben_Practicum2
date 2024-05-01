clear all
close all
clc

load("DatasetCV.mat")

max_n = 12;
CVn_LOO_mem = zeros(max_n, 1);

CVn_mem = zeros(length(x), 1);
for a = 1:length(x)
        x_a = [x(1:a-1); x(a+1:end)];
        y_a = [y(1:a-1); y(a+1:end)];
        cat_a = [cat(1:a-1); cat(a+1:end)];

        B = cat_a;
        A = ones(length(x)-1, 2);

        mdl = fitclinear(A, B, "Learner", "logistic");

        A_a = ones(1, 2);
        B_a = cat(a);

        voorspel_a = predict(mdl, A_a);
        CVn = (voorspel_a ~= B_a);
        CVn_mem(a) = CVn;
end
CVn_LOO = (1/length(x)) * sum(CVn_mem);
CVn_LOO_mem(1) = CVn_LOO;

for n = 1:max_n
    % we halen telkens 1 waarde uit de verzameling en trainen het model op
    % de rest
    CVn_mem = zeros(length(x), 1);
    for a = 1:length(x)
        x_a = [x(1:a-1); x(a+1:end)];
        y_a = [y(1:a-1); y(a+1:end)];
        cat_a = [cat(1:a-1); cat(a+1:end)];

        B = cat_a;

        A = zeros(length(x)-1, 2*n);
        for i = 1:n
            A(:, 2*i-1) = x_a.^i;
            A(:, 2*i) = y_a.^i;
        end

        mdl = fitclinear(A, B, "Learner", "logistic");

        A_a = zeros(1, 2*n);
        for i = 1:n
            A_a(:, 2*i-1) = x(a).^i;
            A_a(:, 2*i) = y(a).^i;
        end

        B_a = cat(a);

        voorspel_a = predict(mdl, A_a);
        CVn = (voorspel_a ~= B_a);
        CVn_mem(a) = CVn;
    end

    CVn_LOO = sum(CVn_mem) / length(x);
    CVn_LOO_mem(n+1) = CVn_LOO;

end

figure
plot(0:max_n, CVn_LOO_mem, "*");
xlabel("n");
ylabel("CVn_LOO")
grid on
title("kruisvalidatiefout voor LOOCV")
