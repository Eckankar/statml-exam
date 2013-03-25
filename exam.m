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
hiddenN = 10; % number of hidden neurons
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
[Xs, Ys] = meshgrid(1:1:starTrainN,1:1:starTrainN);
mask = logical(1 - eye(starTrainN));
coords = [Xs(mask) Ys(mask)];

%% Calculate Jaakkola sigma
coordN = size(coords, 1);
jaak_sigma = median(arrayfun(@(i) norm(starTrainP(coords(i,1)) - starTrainP(coords(i,2))), 1:1:coordN));

%% Question 4 contd.
b = exp(1);

cv_indices = crossvalind('Kfold', starTrainN, 5);
cp = classperf(starTrainC);

correct_rates = zeros(5,7);

for i = -1:1:3
    C = b^i;
    for j = -3:1:3
        sigma = jaak_sigma * b^j;
        
        for k = 1:5
            testI = (cv_indices == k); trainI = ~testI;
            
            model = svmtrain(starTrainP(trainI,:), starTrainC(trainI), ...
                          'boxconstraint', C,  ...
                          'kernel_function', 'rbf', ...
                          'rbf_sigma', sigma);
            prediction = svmclassify(model, starTrainP(testI,:));
            cp = classperf(cp, prediction, testI);
        end
        
        correct_rates(i+2,j+4) = cp.CorrectRate;
    end
end