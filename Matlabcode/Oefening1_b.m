clear all
close all
clc

load('DatasetCV.mat');

disp(find(abs(x + 0.28649) < 0.0001))

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
figure;
gscatter(x, y, B, 'rb');
hold on
contour(X, Y, Z, [0.5, 0.5], 'k');
xlabel('X');
ylabel('Y');
title(['Model voor n = ', num2str(0), ' Verkeerd geclassificeerd: ', num2str(misclassified)]);
legend('Klasse -1', 'Klasse 1', 'Scheidingslijn');

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
    figure;
    gscatter(x, y, B, 'rb');
    hold on
    contour(X, Y, Z, [0.5, 0.5], 'k');
    xlabel('X');
    ylabel('Y');
    title(['Model voor n = ', num2str(n), '; Verkeerd geclassificeerd: ', num2str(misclassified)]);
    legend('Groep -1', 'Groep 1', 'scheidingswand');
end
