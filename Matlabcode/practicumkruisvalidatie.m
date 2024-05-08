%% b

load('DatasetCV.mat');

max_n = 5;

B = cat;

%voor n = 0
A = ones(size(x, 1), 2);

mdl = fitclinear(A, B, "Learner", "logistic");

predicted = predict(mdl, A);
misclassified = sum(predicted ~= B);

xRange = linspace(min(x), max(x), 100);
yRange = linspace(min(y), max(y), 100);
[X, Y] = meshgrid(xRange, yRange);
Z = zeros(size(X));

for i = 1:numel(xRange)
    for j = 1:numel(yRange)
        % Voorspel de klasse voor dit punt
        Z(j, i) = predict(mdl, ones(1, 2));
    end
end

%plot de resultaten
figure
gscatter(x, y, B, 'br');
hold on
contour(X, Y, Z, [0.5, 0.5], 'k');
xlabel('X');
ylabel('Y');
title(['Model voor n = ', num2str(0), ' Verkeerd geclassificeerd: ', num2str(misclassified)]);
legend('Klasse 1', 'Klasse -1', 'Scheidingslijn');

% voor n = 1:5
for n = 1:max_n
    % Bouw de matrix A
    A = zeros(size(x, 1), 2*n);
    for i = 1:n
        A(:, 2*i-1) = x.^i;
        A(:, 2*i) = y.^i;
    end
    
    % Train het logistische regressiemodel
    mdl = fitclinear(A, B, 'Learner', 'logistic');

    predicted = predict(mdl, A);
    misclassified = sum(predicted ~= B);

    % Genereer een grid voor voorspellingen
    xRange = linspace(min(x), max(x), 100);
    yRange = linspace(min(y), max(y), 100);
    [X, Y] = meshgrid(xRange, yRange);
    Z = zeros(size(X));

    % Voorspel klassenlabels voor elk punt in het grid
    for i = 1:numel(xRange)
        for j = 1:numel(yRange)
            % Construeer de features voor voorspelling
            point_features = zeros(1, 2*n);
            for k = 1:n
                point_features(2*k-1) = xRange(i)^k;
                point_features(2*k) = yRange(j)^k;
            end
            % Voorspel de klasse voor dit punt
            Z(j, i) = predict(mdl, point_features);
        end
    end

    % Plot de resultaten
    figure
    gscatter(x, y, B, 'br');
    hold on
    contour(X, Y, Z, [0.5, 0.5], 'k');
    xlabel('X');
    ylabel('Y');
    title(['Model voor n = ', num2str(n), '; Verkeerd geclassificeerd: ', num2str(misclassified)]);
    legend('Groep 1', 'Groep -1', 'scheidingswand');
end

%% c

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

%% d

load("DatasetCV.mat");

helft_lengte = floor(length(x)/2);

figure

for k = 1:6
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
        fout_class = sum(voorspel_groep2 ~= B_2);
        
        CVn = fout_class / helft_lengte;
        CVn_mem(n+1) = CVn;
    end
    
    %plot de resultaten
    
    subplot(3, 2, k)
    plot(0:20, CVn_mem, "r*");
    xlabel('n')
    ylabel('CV_n')
    grid on
    title('kruisvalidatiefout')
end

%% e

load("DatasetCV.mat")

max_n = 12;
CVn_LOO_mem = zeros(max_n+1, 1);

%berekening voor n = 0
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
CVn_LOO = sum(CVn_mem)/length(x);
CVn_LOO_mem(1) = CVn_LOO;

% berekening voor n = 1:12
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
plot(0:max_n, CVn_LOO_mem, "r*");
xlabel("n");
ylabel("CVn_LOO")
grid on
title("kruisvalidatiefout voor LOOCV")

%% f

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
plot(0:max_n, CVn_k_mem, "r*");
xlabel("n");
ylabel("CVn_k")
grid on
title("kruisvalidatiefout voor K-voudig")

%% g

load('DatasetCV.mat')

K = 10;
max_n = 20;

figure
for j = 1:6
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
    
    % Berekeningen voor n = 1:20 en K = 10
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
    
    subplot(3, 2, j)
    plot(0:max_n, CVn_k_mem, "*");
    xlabel("n");
    ylabel("CVn_k")
    grid on
    title("kruisvalidatiefout voor K-voudig")

    [min_value, min_index] = min(CVn_k_mem);

    hold on
    plot(min_index-1, min_value, 'r*');
    hold off

end

%% h

load("BigDatasetCV.mat")

n = 3;

N_array = 10:20:390;

CVn_LOO_mem = zeros(length(N_array), 1);
index = 1;

for N = 10:20:390

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

    CVn_LOO = sum(CVn_mem) / length(xN);
    CVn_LOO_mem(index) = CVn_LOO;
    index = index + 1;
end

figure(1)
plot(N_array', CVn_LOO_mem, "*");
xlabel("N");
ylabel("CVn_LOO")
grid on
title("kruisvalidatiefout voor LOOCV")