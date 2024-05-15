clear all; close all; clc

load('BigDatasetCV.mat')

K_array = 2:15;
n = 3;
number_of_samples = 200;
N = 400;

CVn_k_average_mem = zeros(length(K_array), 1);
CVn_k_variance_mem = zeros(length(K_array), 1);
% bereken het gemiddelde voor K = 1

% bereken de gemiddeldes voor K = 2:15
for K = K_array
    disp(newline + "the value of K is: " + K)

    CVn_k_mem = zeros(number_of_samples, 1);

    nbytes = fprintf('processing sample 0 of %d', length(1:number_of_samples));
    for sample = 1:number_of_samples

        while nbytes > 0
            fprintf('\b')
            nbytes = nbytes - 1;
        end
        nbytes = fprintf('processing sample %d of %d', sample, length(1:number_of_samples));

        % bepaal een random volgorde van indices
        random_indices = randperm(length(x), N);

        CVn_mem = zeros(K, 1);
        for k = 1:K
            % Bepaal de indeces voor de K-de groep
            start_kInt = round(1+(N/K)*(k-1));
            end_kInt = round((N/K)*k);
    
            % Bepaal welke indeces in de trainingset zitten en welke in de test
            indices_k = [random_indices(1:start_kInt-1), random_indices(end_kInt+1:end)];
            indices_test = random_indices(start_kInt:end_kInt); 

            x_k = x(indices_k);
            y_k = y(indices_k);
            cat_k = cat(indices_k);
                
            B_k = cat_k;

            % Bouw matrix A
            A_k = zeros(length(x_k), 2*n);
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
        
            A_test = zeros(length(x_test), 2*n);
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
        CVn_k_mem(sample) = CVn_k;
    end
    CVn_k_average = mean(CVn_k_mem);
    CVn_k_average_mem(K) = CVn_k_average;
    % for 1_j
    CVn_k_variance = var(CVn_k_mem);
    CVn_k_variance_mem(K) = CVn_k_variance;
end

figure
plot(2:15, CVn_k_average_mem(2:15), "r*");
xlabel("K");
ylabel("average CVn_k")
grid on
title("gemiddelde kruisvalidatiefout")

figure
plot(2:15, CVn_k_variance_mem(2:15), "r*");
xlabel("K");
ylabel("variance of CVn_k")
grid on
title("gemiddelde kruisvalidatiefout")