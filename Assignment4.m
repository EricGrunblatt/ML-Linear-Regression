% Assignment 4: Linear Regression, Logistic Regression, Stochastic Gradient Descent, Binary Classification
% Eric Grunblatt

% Read X and Y to use throughout the assignment
X = dlmread('X.txt'); %X: (1+d)*N; d=2 in this dataset for visualization; N=20 in this dataset
Y = dlmread('Y.txt'); %Y: 1*N
numRows = size(X,1);
numCols = size(X,2);

%%%%%% PART 1: LINEAR REGRESSION %%%%%%
% Implement Linear Regression algorithm
w_LinearRegression = (transpose(X) * X) \ (transpose(X) * Y);

% Apply w_LinearRegression to data set and compute error rate
error_w_linreg = 0;
for i=1:numRows
    if(sign(X(i,:) * w_LinearRegression) ~= sign(Y(i)))
        error_w_linreg = error_w_linreg + 1;
    end
end
fprintf('Linear Regression Error Rate: %f\n', (error_w_linreg / numRows));

% Using w_LinearRegression as the initialization to w_PLA, compare to w_PLA = 0
w_LinearRegression = transpose(w_LinearRegression);
errors_w_lr = 0;
total_w_lr = 0;
currentErrors_w_lr = 1;
random = randperm(numRows);
while(currentErrors_w_lr ~= 0)
    currentErrors_w_lr = 0;
    for a=1:numRows % Iterating through each vector
        i = random(a);
        currentNum_w_lr = w_LinearRegression * transpose(X(i,:));
        % Identifying whether there is an error or not, if so, correct it and
        % add to the total number of errors
        total_w_lr = total_w_lr + 1;
        if(sign(currentNum_w_lr) ~= sign(Y(i)))
            currentErrors_w_lr = currentErrors_w_lr + 1;
            w_LinearRegression = w_LinearRegression + Y(i)*X(i,:); % If error, add to w_LinearRegression
        end
    end
    errors_w_lr = errors_w_lr + currentErrors_w_lr;
end

% Running PLA algorithm where w_PLA = 0
weights = zeros(1, numCols);
errors_w = 0;
total_w = 0;
currentErrors_w = 1;
while(currentErrors_w ~= 0)
    currentErrors_w = 0;
    for a=1:numRows % Iterating through each vector
        i = random(a);
        currentNum_w = weights * transpose(X(i,:));
        % Identifying whether there is an error or not, if so, correct it and
        % add to the total number of errors
        total_w = total_w + 1;
        if(sign(currentNum_w) ~= sign(Y(i)))
            currentErrors_w = currentErrors_w + 1;
            weights = weights + Y(i)*X(i,:); % If error, add to weights
        end     
    end
    errors_w = errors_w + currentErrors_w;   
end

% Plotting Points
figure;
for i=1:numRows
    if(Y(i) == 1) % Y is shown as 1, then it is greater than threshold
        plot(X(i,2), X(i,3), 'o', 'color', 'blue');
        hold on
    else % Y is shown as -1, and is less than the threshold
        plot(X(i,2), X(i,3), 'x', 'color', 'red');
        hold on
    end
end
% Creating/Plotting boundary line
slope_w = -weights(2)/weights(3);
slope_w_lr = -w_LinearRegression(2)/w_LinearRegression(3);
%y =mx+c, m is slope and c is intercept(0)
x = [min(X(:,2)),max(X(:,2))];
y_w = (slope_w*x);
y_w_lr = (slope_w_lr*x);
line(x, y_w, 'color', 'green');
line(x,y_w_lr, 'color', 'black');
hold off
% Report error rate
fprintf('w_0 = 0 PLA Error Rate: %f, Total Iterations: %f\n', (errors_w/total_w), total_w);
fprintf('w_0 = w_LinearRegression PLA Error Rate: %f, Total Iterations: %f\n', (errors_w_lr/total_w_lr), total_w_lr);



%%%%%% PART 2: LOGISTIC REGRESSION %%%%%%
% Implement Logistic Regression algorithm
alpha = 1.5;
w_LogisticRegression = zeros(1, numCols);
epsilon = 0.01;
totalIterations = 0;
while(totalIterations < 100)
    E_w = zeros(1, numCols);
    totalIterations = totalIterations + 1;
    for i=1:numRows
        x = (-Y(i) * w_LogisticRegression * transpose(X(i,:)));
        e = exp(x);
        sig = e/(e+1);
        E_w = E_w + sig * (Y(i) * X(i,:));
    end
    E_w = E_w / numRows;
    w_LogisticRegression = w_LogisticRegression + (alpha * E_w);
    if((E_w * transpose(E_w)) < epsilon)
        break;
    end
end
% Applying w_LogisticRegression to the dataset for binary classification error rate
error_w_logreg = 0;
for i=1:numRows
    if(sign(X(i,:) * transpose(w_LogisticRegression)) ~= sign(Y(i)))
        error_w_logreg = error_w_logreg + 1;
    end
end
fprintf('w_0 = 0 Logistic Regression Error Rate: %f\n', (error_w_logreg / numRows));

% Implement Logistic Regression algorithm with w_0 set as w_LinearRegression
w_LogisticRegression = transpose((transpose(X) * X) \ (transpose(X) * Y));
epsilon = 0.01;
totalIterations = 0;
while(totalIterations < 100)
    E_w = zeros(1, numCols);
    totalIterations = totalIterations + 1;
    for i=1:numRows
        x = (-Y(i) * w_LogisticRegression * transpose(X(i,:)));
        e = exp(x);
        sig = e/(e+1);
        E_w = E_w + (sig * (Y(i) * X(i,:)));
    end
    E_w = E_w / numRows;
    w_LogisticRegression = w_LogisticRegression + (alpha * E_w);
    if(sqrt(E_w * transpose(E_w)) < epsilon)
        break;
    end
end
% Applying w_LogisticRegression to the dataset for binary classification error rate
error_w_logreg = 0;
for i=1:numRows
    if(sign(X(i,:) * transpose(w_LogisticRegression)) ~= sign(Y(i)))
        error_w_logreg = error_w_logreg + 1;
    end
end
fprintf('w_0 = w_LinearRegression Logistic Regression Error Rate: %f\n', (error_w_logreg / numRows));


%%%%%% PART 3: SGD LOGISTIC REGRESSION %%%%%%
% Implement Logistic Regression algorithm with Stochastic Gradient Descent(SGD)
w_sgd = zeros(1, numCols);
K = 25;
epsilon = 0.01;
E = E/K;
totalIterations = 0;
random = randperm(numRows);
while(totalIterations < 100)
    E_w = zeros(1, numCols);
    totalIterations = totalIterations + 1;
    for a=1:K
        i = random(a);
        x = (-Y(i) * w_sgd * transpose(X(i,:)));
        e = exp(x);
        sig = e/(e+1);
        E_w = E_w + (sig * (Y(i) * X(i,:)));
    end
    E_w = E_w / K;
    w_sgd = w_sgd + (alpha * E_w);
    if(sqrt(E_w * transpose(E_w)) < epsilon)
        break;
    end
end
% Applying w_LogisticRegression to the dataset for binary classification error rate
error_w_sgd = 0;
for i=1:numRows
    if(sign(X(i,:) * transpose(w_sgd)) ~= sign(Y(i)))
        error_w_sgd = error_w_sgd + 1;
    end
end
fprintf('w_0 = 0 Logistic Regression (SGD) Error Rate: %f\n', (error_w_sgd / numRows));