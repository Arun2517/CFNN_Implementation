clc;
clear;
close all;

%% Load Dataset

data = readtable('/Users/arun/Desktop/Soman Sir Notes/latest_neural_breast cancer/wdbc.data.csv');

%% Features (10 selected features)

X = table2array(data(:,1:10));

%% Class Labels

Y = data.class;

%% Normalize Features

X = normalize(X);

%% One Hot Encoding

YOH = full(ind2vec(Y'+1))';

%% 80-20 Train Test Split

cv = cvpartition(Y,'HoldOut',0.2);

XTrain = X(training(cv),:);
YTrain = YOH(training(cv),:);

XTest = X(test(cv),:);
YTest = Y(test(cv));

fprintf('\n');
fprintf('=========================================\n');
fprintf(' SINGLE HIDDEN LAYER PSEUDOINVERSE\n');
fprintf('=========================================\n');

neurons = [10 20 50 100 200];

for hidden = neurons

    rng(1);

    %% Random Matrices

    Q1 = randn(size(XTrain,2),hidden);

    U1 = randn(size(YTrain,2),hidden);

    %% Hidden Target

    Z1_tilde = tanh(XTrain*Q1) + tanh(YTrain*U1);

    %% Hidden Weight

    W1 = pinv(XTrain)*Z1_tilde;

    %% Hidden Layer Output

    Z1 = tanh(XTrain*W1);

    %% Output Weight

    W0 = pinv(Z1)*YTrain;

    %% Training Prediction

    YPredTrain = Z1*W0;

    [~,trainClass] = max(YPredTrain,[],2);
    trainClass = trainClass - 1;

    trainAcc = mean(trainClass == Y(training(cv)))*100;

    %% Testing

    Z1Test = tanh(XTest*W1);

    YPredTest = Z1Test*W0;

    [~,testClass] = max(YPredTest,[],2);
    testClass = testClass - 1;

    testAcc = mean(testClass == YTest)*100;

    fprintf('Neurons = %3d | Train = %6.2f%% | Test = %6.2f%%\n',...
        hidden,trainAcc,testAcc);

end

fprintf('=========================================\n');

%% TWO HIDDEN LAYERS

fprintf('\n');
fprintf('=========================================\n');
fprintf(' TWO HIDDEN LAYER PSEUDOINVERSE\n');
fprintf('=========================================\n');

architectures = {
    [20 20]
    [20 50]
    [50 50]
    [50 100]
    [100 100]
};

for k = 1:length(architectures)

    arch = architectures{k};

    h1 = arch(1);
    h2 = arch(2);

    rng(1);

    %% Layer 1

    Q1 = randn(size(XTrain,2),h1);
    U1 = randn(size(YTrain,2),h1);

    Z1_tilde = tanh(XTrain*Q1) + tanh(YTrain*U1);

    W1 = pinv(XTrain)*Z1_tilde;

    Z1 = tanh(XTrain*W1);

    %% Layer 2

    Q2 = randn(h1,h2);
    U2 = randn(size(YTrain,2),h2);

    Z2_tilde = tanh(Z1*Q2) + tanh(YTrain*U2);

    W2 = pinv(Z1)*Z2_tilde;

    Z2 = tanh(Z1*W2);

    %% Output Layer

    W0 = pinv(Z2)*YTrain;

    %% Training Accuracy

    YPredTrain = Z2*W0;

    [~,trainClass] = max(YPredTrain,[],2);
    trainClass = trainClass - 1;

    trainAcc = mean(trainClass == Y(training(cv)))*100;

    %% Testing

    Z1Test = tanh(XTest*W1);

    Z2Test = tanh(Z1Test*W2);

    YPredTest = Z2Test*W0;

    [~,testClass] = max(YPredTest,[],2);
    testClass = testClass - 1;

    testAcc = mean(testClass == YTest)*100;

    fprintf('Architecture = [%d %d] | Train = %6.2f%% | Test = %6.2f%%\n',...
        h1,h2,trainAcc,testAcc);

end

fprintf('=========================================\n');

%% Best Architecture Example

h1 = 50;
h2 = 50;

Q1 = randn(size(XTrain,2),h1);
U1 = randn(size(YTrain,2),h1);

Z1_tilde = tanh(XTrain*Q1) + tanh(YTrain*U1);

W1 = pinv(XTrain)*Z1_tilde;

Z1 = tanh(XTrain*W1);

Q2 = randn(h1,h2);
U2 = randn(size(YTrain,2),h2);

Z2_tilde = tanh(Z1*Q2) + tanh(YTrain*U2);

W2 = pinv(Z1)*Z2_tilde;

Z2 = tanh(Z1*W2);

W0 = pinv(Z2)*YTrain;

%% Final Test

Z1Test = tanh(XTest*W1);
Z2Test = tanh(Z1Test*W2);

YPred = Z2Test*W0;

[~,predClass] = max(YPred,[],2);
predClass = predClass - 1;

accuracy = mean(predClass == YTest)*100;

fprintf('\n');
fprintf('Final Test Accuracy = %.2f%%\n',accuracy);
size(XTrain)
size(Q1)
size(U1)
size(Z1_tilde)
size(W1)
size(Z1)
size(W0)