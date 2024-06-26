clear all
close all
clc

load("BigDatasetCV.mat")

n = 3;

N_array = 10:20:390;
CVn_LOO_average_mem = zeros(length(N_array), 1);
CVn_LOO_variance_mem = zeros(length(N_array), 1);

index = 1;
number_of_samples = 50;

for N = 10:20:390

    disp(newline + "the value of N is: " + N)
    
    CVn_LOO_mem = zeros(number_of_samples, 1);

    nbytes = fprintf('processing sample 0 of %d', length(1:number_of_samples));

    for sample = 1:number_of_samples
        
        while nbytes > 0
            fprintf('\b')
            nbytes = nbytes - 1;
        end
        nbytes = fprintf('processing sample %d of %d', sample, length(1:number_of_samples));


        random_indices = randperm(length(x));
    
        xN = x(random_indices(1:N));
        yN = y(random_indices(1:N));
        catN = cat(random_indices(1:N));
    
    
        
        CVn_mem = zeros(length(xN), 1);
        for a = 1:length(xN)
            xN_a = [xN(1:a-1); xN(a+1:end)];
            yN_a = [yN(1:a-1); yN(a+1:end)];
            catN_a = [catN(1:a-1); catN(a+1:end)];
    
            B = catN_a;
    
            A = zeros(length(xN)-1, 2*n);
            for i = 1:n
                A(:, 2*i-1) = xN_a.^i;
                A(:, 2*i) = yN_a.^i;
            end
    
            mdl = fitclinear(A, B, "Learner", "logistic");
    
            A_a = zeros(1, 2*n);
            for i = 1:n
                A_a(:, 2*i-1) = xN(a).^i;
                A_a(:, 2*i) = yN(a).^i;
            end
    
            B_a = catN(a);
    
            voorspel_a = predict(mdl, A_a);
            CVn = (voorspel_a ~= B_a);
            CVn_mem(a) = CVn;
        end
    
        CVn_LOO = mean(CVn_mem);
        CVn_LOO_mem(sample) = CVn_LOO;
    end
    %calculate the average
    CVn_LOO_average = mean(CVn_LOO_mem);
    %Calculate the variance
    CVn_LOO_variance = var(CVn_LOO_mem);
    CVn_LOO_average_mem(index) = CVn_LOO_average;
    CVn_LOO_variance_mem(index) = CVn_LOO_variance;

    index = index + 1;
end

figure(1)
plot(N_array', CVn_LOO_average_mem, "*");
xlabel("N");
ylabel("Gemiddelde CVn_LOO")
grid on
title("Gemiddelde kruisvalidatiefout voor LOOCV")
legend('Gemiddelde kruisvalidatiefout')

figure(2)
plot(N_array', CVn_LOO_variance_mem, "*");
xlabel("N");
ylabel("variance of CVn_LOO")
grid on
title("kruisvalidatiefout voor LOOCV")
