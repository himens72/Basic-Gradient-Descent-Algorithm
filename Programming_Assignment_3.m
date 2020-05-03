% References
% https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
% https://www.mathworks.com/help/matlab/matlab_prog/floating-point-numbers.html#f2-98670
% https://www.mathworks.com/help/matlab/ref/norm.html
% https://www.mathworks.com/help/matlab/ref/break.html
% https://www.mathworks.com/help/matlab/ref/min.html#bupsfdw
% mathworks.com/matlabcentral/answers/75568-how-can-i-normalize-data-between-0-and-1-i-want-to-use-logsig
% https://www.mathworks.com/help/matlab/ref/numel.html
% https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e
% https://stats.stackexchange.com/questions/137834/clarification-about-perceptron-rule-vs-gradient-descent-vs-stochastic-gradient
% https://www.mathworks.com/matlabcentral/fileexchange/63046-perceptron-learning
% https://www.mathworks.com/matlabcentral/fileexchange/63046-perceptron-learning
% https://www.mathworks.com/matlabcentral/fileexchange/27754-rosenblatt-s-perceptron
% https://www.mathworks.com/matlabcentral/answers/419270-simple-perceptron-algorithm-in-matlab-cannot-draw-the-classifier-line
% https://www.mathworks.com/matlabcentral/answers/197240-problem-while-implementing-gradient-descent-algorithm-in-matlab
% https://www.mathworks.com/matlabcentral/fileexchange/35535-simplified-gradient-descent-optimization
% https://stackoverflow.com/questions/21799435/gradient-descent-matlab-implementation
% https://www.codeproject.com/Articles/879043/Implementing-Gradient-Descent-to-Solve-a-Linear-Re
clc;
load samples.mat;
o = ones(10);
o = o(: , 1);
learning_rate = 0.1;
iteration = 200;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X12( : , 1) = [samples( :  , 1 );samples( :  , 3 )];
X12( : , 2) = [samples( :  , 2 );samples( :  , 4 )];
X12( : , 3) = [1 * o; 2 * o];
Y12 = [1 * o; 2 * o];
[W, bias, converged, history] = Perceptron(X12', Y12' ,learning_rate,iteration + 1);
figure, plot(history,'-o');
hold on;
xlabel('Iteration Number');
ylabel('Criterion Function');
grid on;
title('Question A : Perceptron Criterion Function for W1 and W2','fontsize', 13);
fprintf('Question A :W for  Perceptron Criterion Function for W1 and W2  = ');
disp(W');
BP12 = numel(history);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X14( : , 1) = [samples( :  , 1 );samples( :  , 7 )];
X14( : , 2) = [samples( :  , 2 );samples( :  , 8 )];
X14( : , 3) = [1 * o; 2 * o];
Y14 = X14( : , 3);
tic;
[W, bias, converged, history] = Perceptron(X14', Y14' ,learning_rate,iteration + 1);
time = toc;
disp(time)
figure, plot(history,'-o');
hold on;
xlabel('Iteration Number');
ylabel('Criterion Function');
grid on;
title('Question A : Perceptron Criterion Function for W1 and W4','fontsize', 13);
fprintf('Question A :W for  Perceptron Criterion Function for W1 and W4  = ');
disp(W');
BP14 = numel(history);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X12( : , 1) = [1 * o; 1 * o];
X12( : , 2) = [samples( :  , 1 );samples( :  , 3 )];
X12( : , 3) = [samples( :  , 2 );samples( :  , 4 )];
Y12 = [1 * o; 2 * o];
minimum = min(min(X12));
maximum = max(max(X12));
XS = (X12 -  minimum) / ( maximum - minimum);
[W, converged, history] = Gradient(XS, Y12, 0.1, 0.02,500);
figure, plot(history,'-');
xlim([-20 500]);
hold on;
xlabel('Iteration Number');
ylabel('Loss Function');
grid on;
title('Question A : Gradient Descent for W1 and W2','fontsize', 13);
fprintf('Question A :W for  Gradient Descent for W1 and W2  = ');
disp(W(1:2 , :)');
GD12 = numel(history);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X14( : , 1) = [1 * o; 1 * o];
X14( : , 2) = [samples( :  , 1 );samples( :  , 3 )];
X14( : , 3) = [samples( :  , 7 );samples( :  , 8 )];
Y14 = [1 * o; 2 * o];
minimum = min(min(X14));
maximum = max(max(X14));
XS = (X14 -  minimum) / ( maximum - minimum);
[W, converged, history] = Gradient(XS, Y14, 0.1, 0.02,500);
figure, plot(history,'-');
xlim([-20 500]);
hold on;
xlabel('Iteration Number');
ylabel('Loss Function');
grid on;
title('Question A : Gradient Descent for W1 and W4','fontsize', 13);
fprintf('Question A :W for  Gradient Descent for W1 and W4  = ');
disp(W(1:2 , :)');
GD14 = numel(history);

fprintf('Question B :Iteration Count for Perceptron W1 and W2  = ');
disp(BP12);
fprintf('Question B :Iteration Count for Perceptron W1 and W4  = ');
disp(BP14);
fprintf('Question B :Iteration Count for Gradient Descent W1 and W2  = More than ');
disp(GD12);
fprintf('Question B :Iteration Count for Gradient Descent W1 and W4  = More than ');
disp(GD14);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[PLearning12, PTime12] = convergence_time(samples( :  , 1),samples( :  , 2 ),samples( :  , 3 ),samples( :  ,  4),o);
figure, plot(PLearning12,PTime12,'-o');
hold on;
xlabel('Learning rate');
ylabel('Convergence Time');
grid on;
title('Question C : Perceptron Criterion Function for W1 and W2','fontsize', 8);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[PLearning12, PTime12] = convergence_time(samples( :  , 1),samples( :  , 2 ),samples( :  , 7 ),samples( :  ,  8),o);
figure, plot(PLearning12,PTime12,'-o');
hold on;
xlabel('Learning rate');
ylabel('Convergence Time');
grid on;
title('Question C : Perceptron Criterion Function for W1 and W4','fontsize', 8);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[PLearning12, PTime12] = convergence_time1(samples( :  , 1),samples( :  , 2 ),samples( :  , 3 ),samples( :  ,  4),o);
figure, plot(PLearning12,PTime12,'-o');
hold on;
xlabel('Learning rate');
ylabel('Convergence Time');
grid on;
title('Question C : Gradient Descent for W1 and W2','fontsize', 8);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[PLearning12, PTime12] = convergence_time1(samples( :  , 1),samples( :  , 2 ),samples( :  , 7 ),samples( :  ,  8),o);
figure, plot(PLearning12,PTime12,'-o');
hold on;
xlabel('Learning rate');
ylabel('Convergence Time');
grid on;
title('Question C : Gradient Descent for W1 and W4','fontsize', 8);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W, converged, history] = Gradient(X, Y,  learning_rate, tolerance, iteration)
[n_samples, dimension] = size(X);
i = 1;
W = [0;0;0];
history = [];
converged = false;
while i <= iteration
    W = W - (learning_rate / n_samples) * X' * (X * W - Y);
    % disp("Iteration");
    %disp(i);
    %disp("W is");
    % disp(W);
    cost = (1.0 / (2.0 * n_samples)) * (X * W - Y)' * (X * W - Y);
    %disp("\nCost is");
    % disp(cost);
    % disp("\nHistory is");
    history = [history double(cost)];
    %disp(history);
    if norm(history) <= tolerance
        break;
    end
    i = i +1;
end
if i <= iteration
    converged = true;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_graph(x1, y1, x2 , y2, x3, y3,w,bias,str,legend1,legend2)
figure, plot(x1,y1,'bo');
hold on;
plot(x2,y2, 'go', 'linewidth', 2);
x = linspace(x3,y3,100);
y = (bias + w(1,1) * x)/ - w(2,1);
plot(x,y,'-');
legend({legend1,legend2},'Location','northeast');
xlabel('X1');
ylabel('X2');
grid on;
title(str, 'fontsize', 18);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, bias, converged, history] = Perceptron(X, Y,  learning_rate, iteration)
o = ones(10);
o = o(1 , :);
[dimensions, n] = size(X');
converged = false;
X(3 , :) = [o, -1 * o];
X(1 : 2, 11: 20) = -1 * X(1 : 2,11: 20);
[dimensions , n] = size(X);
W = [0; 0;0];
i = 1;
history = [];
while (converged == false && i <= iteration)
    val = X' * W;
    temp = val( : , 1);
    idx = find(temp <= 0);
    if length(idx) == 0
        history = [history, 0];
        converged = true;
    else
        history = [history length(idx)];
        W = W + learning_rate * sum( X(:, idx),2 );
    end
    i = i + 1;
end
bias = W(3, :);
W = W(1: 2, :);
end

function [PLearning12, PTime12] = convergence_time(a,b,c,d,o)
iteration = 100;
PTime12 = [];
PLearning12 = [];

X12( : , 1) = [a;b];
X12( : , 2) = [c;d];
X12( : , 3) = [1 * o; 2 * o];
Y12 = [1 * o; 2 * o];
tic
PLearning12 = [PLearning12 0.01];
[W, bias, converged, history] = Perceptron(X12', Y12' ,0.01,iteration + 1);
time = toc;
PTime12 = [PTime12 time];

tic
PLearning12 = [PLearning12 0.05];
[W, bias, converged, history] = Perceptron(X12', Y12' ,0.05,iteration + 1);
time = toc;
PTime12 = [PTime12 time];

tic
PLearning12 = [PLearning12 0.1];
[W, bias, converged, history] = Perceptron(X12', Y12' ,0.1,iteration + 1);
time = toc;
PTime12 = [PTime12 time];

tic
PLearning12 = [PLearning12 0.3];
[W, bias, converged, history] = Perceptron(X12', Y12' ,0.3,iteration + 1);
time = toc;
PTime12 = [PTime12 time];

tic
PLearning12 = [PLearning12 0.5];
[W, bias, converged, history] = Perceptron(X12', Y12' ,0.5,iteration + 1);
time = toc;
PTime12 = [PTime12 time];
tic
PLearning12 = [PLearning12 0.7];
[W, bias, converged, history] = Perceptron(X12', Y12' ,0.7,iteration + 1);
time = toc;
PTime12 = [PTime12 time];
tic
PLearning12 = [PLearning12 1];
[W, bias, converged, history] = Perceptron(X12', Y12' ,1,iteration + 1);
time = toc;
PTime12 = [PTime12 time];
end

function [PLearning12, PTime12] = convergence_time1(a,b,c,d,o)
iteration = 100;
PTime12 = [];
PLearning12 = [];
X14( : , 1) = [1 * o; 1 * o];
X14( : , 2) = [a;b];
X14( : , 3) = [c;d];
Y14 = [1 * o; 2 * o];
minimum = min(min(X14));
maximum = max(max(X14));
XS = (X14 -  minimum) / ( maximum - minimum);
tic
PLearning12 = [PLearning12 0.01];
[W, converged, history] = Gradient(XS, Y14, 0.01, 0.02,500);
time = toc;
PTime12 = [PTime12 time];
tic
PLearning12 = [PLearning12 0.03];
[W, converged, history] = Gradient(XS, Y14, 0.03, 0.02,500);
time = toc;
PTime12 = [PTime12 time];
tic
PLearning12 = [PLearning12 0.05];
[W, converged, history] = Gradient(XS, Y14, 0.05, 0.02,500);
time = toc;
PTime12 = [PTime12 time];
tic
PLearning12 = [PLearning12 0.07];
[W, converged, history] = Gradient(XS, Y14, 0.07, 0.02,500);
time = toc;
PTime12 = [PTime12 time];
tic
PLearning12 = [PLearning12 0.1];
[W, converged, history] = Gradient(XS, Y14, 0.1, 0.02,500);
time = toc;
PTime12 = [PTime12 time];
tic
PLearning12 = [PLearning12 0.3];
[W, converged, history] = Gradient(XS, Y14, 0.3, 0.02,500);
time = toc;
PTime12 = [PTime12 time];
tic
PLearning12 = [PLearning12 0.5];
[W, converged, history] = Gradient(XS, Y14, 0.5, 0.02,500);
time = toc;
PTime12 = [PTime12 time];
tic
PLearning12 = [PLearning12 0.7];
[W, converged, history] = Gradient(XS, Y14, 0.7, 0.02,500);
time = toc;
PTime12 = [PTime12 time];
tic
PLearning12 = [PLearning12 1];
[W, converged, history] = Gradient(XS, Y14, 1, 0.02,500);
time = toc;
PTime12 = [PTime12 time];

end