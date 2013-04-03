%% Case 1
sunTrain = importdata('data/sunspotsTrainStatML.dt');
sunTest = importdata('data/sunspotsTestStatML.dt');

% split into predictor and target variables
sunTrainP = sunTrain(:,1:5);
sunTrainT = sunTrain(:,6);
sunTrainN = size(sunTrain, 1);
sunTestP = sunTest(:,1:5);
sunTestT = sunTest(:,6);
sunTestN = size(sunTest, 1);

%% Question 1
% Compute the maximum likelihood model
designMatrix = [ones(sunTrainN, 1) sunTrainP];
wML = pinv(designMatrix) * sunTrainT;

% Calculate RMS
valMLTrain = arrayfun(@(i) [1, sunTrainP(i,:)] * wML, 1:sunTrainN);
rmsMLTrain = rms(sunTrainT, valMLTrain);

valMLTest = arrayfun(@(i) [1, sunTestP(i,:)] * wML, 1:sunTestN);
rmsMLTest = rms(sunTestT, valMLTest);

%% Question 2
% Neural network used for regression
hiddenN = 5; % number of hidden neurons
trainingMethod = 'trainrp'; % train using RProp

net = fitnet(hiddenN, trainingMethod);
net.trainParam.showWindow = false;

[net, tr] = train(net, sunTrainP', sunTrainT');

% Calculate RMS for the NN on the training data
nnTrainT = net(sunTrainP');
rmsNNTrain = rms(sunTrainT, nnTrainT);

% Calculate RMS for the NN on the testing data
nnTestT = net(sunTestP');
rmsNNTest = rms(sunTestT, nnTestT);

%% Question 3
years = 1916:1:2011;
figure(1), clf
hold on
plot(years, sunTestT, 'r');
plot(years, valMLTest, 'g');
plot(years, nnTestT, 'b');
title('Sunspot predictions on testing data');
legend('Actual values', 'Linear ML model', 'Neural Network (5 neurons)');
hold off

years = 1716:1:1915;
figure(2), clf
hold on
plot(years, sunTrainT, 'r');
plot(years, valMLTrain, 'g');
plot(years, nnTrainT, 'b');
title('Sunspot predictions on training data');
legend('Actual values', 'Linear ML model', 'Neural Network (5 neurons)');
hold off

%% Case 2
starTrain = importdata('data/quasarsStarsStatMLTrain.dt');
starTest = importdata('data/quasarsStarsStatMLTest.dt');

starTrainP = starTrain(:,1:5);
starTrainC = starTrain(:,6);
starTrainN = size(starTrain, 1);
starTestP = starTest(:,1:5);
starTestC = starTest(:,6);
starTestN = size(starTest, 1);

%% Question 4
starsOnly = starTrainP(starTrainC == 0,:);
quasarsOnly = starTrainP(starTrainC == 1,:);

starsOnlyN = size(starsOnly, 1);
quasarsOnlyN = size(quasarsOnly, 1);

[Xs, Ys] = meshgrid(1:1:starsOnlyN,1:1:quasarsOnlyN);

coords = [Xs(:) Ys(:)];

%% Calculate Jaakkola sigma 
coordN = size(coords, 1);
jaak_sigma = median(arrayfun(@(i) norm(starsOnly(coords(i,1),:) - quasarsOnly(coords(i,2),:)), 1:1:coordN));

%% Question 4 contd.
b = exp(1);

cv_indices = crossvalind('Kfold', starTrainN, 5);
cp_train = classperf(starTrainC);

correct_rates_train = zeros(5,7);

for i = -1:1:3
    C = b^i;
    for j = -3:1:3
        sigma = jaak_sigma * b^(-j/2);
        
        for k = 1:5
            testI = (cv_indices == k); trainI = ~testI;
            
            model = svmtrain(starTrainP(trainI,:), starTrainC(trainI), ...
                          'boxconstraint', C,  ...
                          'kernel_function', 'rbf', ...
                          'rbf_sigma', sigma);
            prediction_train = svmclassify(model, starTrainP(testI,:));
            cp_train = classperf(cp_train, prediction_train, testI);
        end
        
        correct_rates_train(i+2,j+4) = cp_train.CorrectRate;
    end
end

% Extract optimal hyperparameters
[bestI, bestJ] = find(correct_rates_train == max(correct_rates_train(:)));

% Correct for matrix-offsets
bestI = bestI - 2;
bestJ = bestJ - 4;

% Compute optimal C and sigma
bestC = b^bestI;
bestSigma = jaak_sigma * b^(-bestJ/2);

% Train the final SVM on the training set
model = svmtrain(starTrainP, starTrainC, ...
                 'boxconstraint', bestC,  ...
                 'kernel_function', 'rbf', ...
                 'rbf_sigma', bestSigma);
             
% Predict on training data
prediction_train = svmclassify(model, starTrainP);
cp_train = classperf(starTrainC, prediction_train);
correctRateTrain = cp_train.CorrectRate;

% Predict on test data
prediction_test = svmclassify(model, starTestP);
cp_test = classperf(starTestC, prediction_test);
correctRateTest = cp_test.CorrectRate;

%% Case 3
grainTrain = importdata('data/seedsTrain.dt');
grainTest = importdata('data/seedsTest.dt');

grainTrainP = grainTrain(:,1:7);
grainTrainC = grainTrain(:,8);
grainTrainN = size(grainTrain, 1);
grainTestP = grainTest(:,1:7);
grainTestC = grainTest(:,8);
grainTestN = size(grainTest, 1);

%% Question 6
[normGrainTrainP, mu, sigma] = zscore(grainTrainP);
% Normalized data gives a significant eigenvalue to the 3rd component
% Therefore it is not used

[grainPComps, grainScores, grainEvalues] = princomp(grainTrainP);
figure(2), clf
plot(1:size(grainEvalues,1), grainEvalues, 'b')

figure(3), clf
colors = ['r', 'g', 'b'];
hold on
for i = 1:grainTrainN
    color = colors(grainTrainC(i)+1);
    plot(grainScores(i,1), grainScores(i,2), strcat(color, 'x'))
end
hold off

%% Question 7
[grainClusters, clusterCentroids] = kmeans(grainTrainP, 3);

figure(4), clf
hold on
for i = 1:grainTrainN
    color = colors(grainTrainC(i)+1);
    plot(grainScores(i,1), grainScores(i,2), strcat(color, 'x'))
end
for i = 1:3
    c = (clusterCentroids(i,:) - mu) * grainPComps;
    plot(c(1), c(2), 'ko')
end
hold off

%% Question 8

%% Non-linear classification: k-NN
nearestNeighborI = knnsearch(grainTrainP, grainTestP, 'K', 5);
nearestNeighborCs = grainTrainC(nearestNeighborI);
nearestNeighborC = arrayfun(@(i) mode(nearestNeighborCs(i,:)) ,1:1:grainTestN);

figure(5), clf
hold on
for i = 1:grainTestN
    color = colors(nearestNeighborC(i)+1);
    
    projP = grainTestP(i,:) * grainPComps;
    if nearestNeighborC(i) == grainTestC(i)
        marker = 'x';
    else
        marker = 'v';
    end
    
    plot(projP(1), projP(2), strcat(color, marker))
end
hold off

%% Linear classification: LDA
[ldaC, ~, ~, ~, ldaCoeff] = classify(grainTestP, grainTrainP, grainTrainC, 'linear');

figure(6), clf
hold on
for i = 1:grainTestN
    color = colors(ldaC(i)+1);
    
    projP = grainTestP(i,:) * grainPComps;
    if ldaC(i) == grainTestC(i)
        marker = 'x';
    else
        marker = 'v';
    end
    
    plot(projP(1), projP(2), strcat(color, marker))
end
hold off