function [x] = SolveOptProblem(Q, b)
% Function solves the quadratic optimization problem stated to select
% significance and noncollinear features
%
% Input:
% Q - [n, n] - matrix of features similarities
% b - [n, 1] - vector of feature relevances
%
% Output:
% x - [n, 1] - solution of the quadratic optimization problem
%
% Author: Alexandr Katrutsa, 2016 
% E-mail: aleksandr.katrutsa@phystech.edu

[n, ~] = size(Q);
Q = (Q + Q') / 2;

% Use original CVX path when available; otherwise use a toolbox-free
% projected-gradient fallback.
if exist('cvx_begin', 'file') == 2
    try
        cvx_solver Mosek
        cvx_begin
            variable x(n, 1) nonnegative;
            minimize (x'*Q*x - b'*x)
            subject to
                norm(x, 1) <= 1;
        cvx_end
        return
    catch
        % Fall back to projected-gradient solver below.
    end
end

x = ones(n, 1) / n;
L = 2 * norm(Q, 2);
if L <= eps
    L = 1;
end
step = 1 / L;
max_iter = 10000;
tol = 1e-10;

for k = 1:max_iter
    grad = 2 * Q * x - b;
    x_new = proj_l1_ball_nonneg(x - step * grad, 1);
    if norm(x_new - x) <= tol * (1 + norm(x))
        x = x_new;
        break;
    end
    x = x_new;
end
end

function w = proj_l1_ball_nonneg(v, z)
% Project onto {w >= 0, sum(w) <= z}.
v = max(v, 0);
if sum(v) <= z
    w = v;
    return;
end
w = proj_simplex(v, z);
end

function w = proj_simplex(v, z)
% Euclidean projection onto simplex {w >= 0, sum(w) = z}.
u = sort(v, 'descend');
cssv = cumsum(u) - z;
rho = find(u - cssv ./ (1:length(u))' > 0, 1, 'last');
theta = cssv(rho) / rho;
w = max(v - theta, 0);
end

