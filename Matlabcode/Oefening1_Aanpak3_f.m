clear all
close all
clc

load('DatasetCV.mat')

K = 10;
max_n = 20;

CVn_k_mem = zeros(max_n+1, 1);
random_indices = randperm(length(x));

% berekeningen voor n = 0 en K = 10
CVn_mem = zeros(K, 1);
for k = 1:K
    % Bepaal de indeces voor de K-de groep
    start_kInt = 1+(length(x)/K)*(k-1);
    end_kInt = (length(x)/K)*k;
        
    % Bepaal welke indeces in de trainingset zitten en welke in de test
    indices_k = [random_indices(1:start_kInt-1), random_indices(end_kInt+1:end)];
    indices_test = random_indices(start_kInt:end_kInt); 
    
    x_k = x(indices_k);
    y_k = y(indices_k);
    cat_k = cat(indices_k);
        
    B_k = cat_k;
        
    % Bouw matrix A
    A_k = ones(length(x) - length(x)/K, 2);

    mdl = fitclinear(A_k, B_k, "Learner", "logistic");
        
    % test op de testset
    x_test = x(indices_test);
    y_test = y(indices_test);
    cat_test = cat(indices_test);
    B_test = cat_test;

    A_test = ones(length(x)/K, 2);

    voorspel_test = predict(mdl, A_test);
    fout_class = sum(voorspel_test ~= B_test);

    CVn = fout_class / (length(x)/K);
    CVn_mem(k) = CVn;
end

CVn_k = sum(CVn_mem)/K;
CVn_k_mem(1) = CVn_k;

% berekening voor n = 1:20 en K = 10
for n = 1:max_n
    CVn_mem = zeros(K, 1);
    for k = 1:K
        % Bepaal de indeces voor de K-de groep
        start_kInt = 1+(length(x)/K)*(k-1);
        end_kInt = (length(x)/K)*k;
        
        % Bepaal welke indeces in de trainingset zitten en welke in de test
        indices_k = [random_indices(1:start_kInt-1), random_indices(end_kInt+1:end)];
        indices_test = random_indices(start_kInt:end_kInt); 
    
        x_k = x(indices_k);
        y_k = y(indices_k);
        cat_k = cat(indices_k);
        
        B_k = cat_k;
        
        % Bouw matrix A
        A_k = zeros(length(x)-length(x)/K, 2*n);
        for i = 1:n
            A_k(:, 2*i-1) = x_k.^i;
            A_k(:, 2*i) = y_k.^i;  
        end

        mdl = fitclinear(A_k, B_k, "Learner", "logistic");
        
        % test op de testset
        x_test = x(indices_test);
        y_test = y(indices_test);
        cat_test = cat(indices_test);
        B_test = cat_test;

        A_test = zeros(length(x)/K, 2*n);
        for i = 1:n
            A_test(:, 2*i-1) = x_test.^i;
            A_test(:, 2*i) = y_test.^i;
        end

        voorspel_test = predict(mdl, A_test);
        fout_class = sum(voorspel_test ~= B_test);

        CVn = fout_class / (length(x)/K);
        CVn_mem(k) = CVn;
    end
    CVn_k = sum(CVn_mem)/K;
    CVn_k_mem(n+1) = CVn_k;
end

figure
semilogy(0:max_n, CVn_k_mem, "r*");
hold on

% Markeer de laagste waarde in de plot als blauw
[min_value, min_index] = min(CVn_k_mem);
plot(min_index-1, min_value, "b*");
hold off

xlabel("n");
ylabel("CVn_k")
grid on
title("kruisvalidatiefout voor K-voudig")
legend('CVn_k waarden', 'Laagste CVn_k waarde')