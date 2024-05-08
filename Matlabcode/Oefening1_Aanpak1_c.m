clear all
close all
clc

load("DatasetCV.mat");

helft_lengte = floor(length(x)/2);

% bepaal een random verdeling van de indices van de data
random_indices = randperm(length(x));

% de eerste helft van random verdeelde indices is groep 1, de tweede helft
% is groep 2;
eerste_helft_indices = random_indices(1:helft_lengte);
tweede_helft_indices = random_indices(helft_lengte + 1:end);

x_1 = x(eerste_helft_indices);
x_2 = x(tweede_helft_indices);

y_1 = y(eerste_helft_indices);
y_2 = y(tweede_helft_indices);

cat_1 = cat(eerste_helft_indices);
cat_2 = cat(tweede_helft_indices);

% bepaal de modelparameters zoals in 1_b

max_n = 20;
B_1 = cat_1;
B_2 = cat_2;

CVn_mem = zeros(max_n, 1);

% bereken CVn voor n = 0
A_1 = ones(helft_lengte, 2);

mdl = fitclinear(A_1, B_1, "Learner", "logistic");

A_2 = ones(helft_lengte, 2);

voorspel_groep2 = predict(mdl, A_2);
fout_class = sum(voorspel_groep2 ~= B_2);

CVn = fout_class / helft_lengte;
CVn_mem(1) = CVn;

% beren CVn coor n = 1:20
for n = 1:max_n
    % Bouw de matrix A_1
    A_1 = zeros(helft_lengte, 2*n);
    for i = 1:n
        A_1(:, 2*i-1) = x_1.^i;
        A_1(:, 2*i) = y_1.^i;
    end
    
    % Train het logistische regressiemodel
    mdl = fitclinear(A_1, B_1, 'Learner', 'logistic');
    
    % Bouw matrix A_2
    A_2 = zeros(helft_lengte, 2*n);
    for i = 1:n
        A_2(:, 2*i-1) = x_2.^i;
        A_2(:, 2*i) = y_2.^i;
    end
    voorspel_groep2 = predict(mdl, A_2);
    fout_class = sum(voorspel_groep2 ~= B_2)
    
    CVn = fout_class / helft_lengte;
    CVn_mem(n+1) = CVn;
end

%plot de resultaten

figure
plot(0:max_n, CVn_mem, "r*");
xlabel('n')
ylabel('CV_n')
grid on
title('kruisvalidatiefout voor een random gekozen subset')