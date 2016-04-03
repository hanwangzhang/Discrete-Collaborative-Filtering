function [U,V,X,Y] = DCFinit(S, ST, IDX, IDXT, r, alpha, beta, option)
%DCFinit: Initialization for Dicrete Collaborative Filtering as in Eq.(22)

%Input:
%S: user-item score matrix, [m,n] = size(S)
%ST: transpose of ST, for efficient sparse matrix indexing in Matlab, i.e.,
%matlab can only efficiently access sparse matrix by column.
%IDX: nonzero (observed) entry index of S
%IDXT: transpose of IDX for efficient sparse matrix indexing in Matlab.
%r: bit length
%alpha: trade-off paramter. good default = 0.001.
%beta: trade-off paramter. good default = 0.001.
%option:
%option.maxItr: max iterations. Default = 50.
%option.tol: tolerance. Default = 1e-5.
%option.debug: show obj?. Default = false.

%Output:
%U: user vector
%V: item vector
%X: surrogate user vector
%Y: surrogate item vector

%Reference:
%   Hanwang Zhang, Fumin Shen, Wei Liu, Xiangnan He, Huanbo Luan, Tat-seng
%   Chua. "Discrete Collaborative Filtering", SIGIR 2016

%Version: 1.0
%Written by Hanwang Zhang (hanwangzhang AT gmail.com)

maxS = max(max(S));
minS = min(min(S));
[m,n] = size(S);
U = rand(r,m);
V = rand(r,n);
X = UpdateSVD(U);
Y = UpdateSVD(V);
if isfield(option,'maxItr')
    maxItr = option.maxItr;
else
    maxItr = 50;
end
if isfield(option,'tol')
    tol = option.tol;
else
    tol = 1e-5;
end
if isfield(option,'debug')
    debug = option.debug;
else
    debug = false;
end
converge = false;
it = 1;


if debug
    disp(DCFinitObj(S, IDX, maxS, minS,U, V, X, Y, alpha, beta));
end

while ~converge
    tic;
    U0 = U;
    V0 = V;
    X0 = X;
    Y0 = Y;
    parfor i = 1:m
        Vi = V(:,nonzeros(IDXT(:,i)));
        Si = nonzeros(ST(:,i));
        if isempty(Si)
            continue;
        end
        Q = Vi*Vi'+alpha*length(Si)*eye(r);
        L = Vi*Si+2*alpha*X(:,i);
        U(:,i) = Q\L;
    end
    parfor j = 1:n
        Uj = U(:,nonzeros(IDX(:,j)));
        Sj = nonzeros(S(:,j));
        if isempty(Sj)
            continue;
        end
        Q = Uj*Uj'+beta*length(Sj)*eye(r);%quadratic term
        L = Uj*Sj+2*beta*Y(:,j);% linear term
        V(:,j) = Q\L;
    end
    
    
    X = UpdateSVD(U);
    Y = UpdateSVD(V);
    
    toc;
    disp(['DCFinit Iteration:',int2str(it-1)]);
    if it >= maxItr || max([norm(U-U0,'fro') norm(V-V0,'fro') norm(X-X0,'fro') norm(Y-Y0,'fro')]) < max([m n])*tol
        converge = true;
    end
    
    if debug
        disp(DCFinitObj(S,IDX,maxS, minS,U,V,X,Y,alpha,beta));
    end
    it = it+1;
end
end


function obj = DCFinitObj(S, IDX, maxS, minS, U, V, X, Y, alpha, beta)
[~,n] = size(S);
r = size(U,1);
loss = zeros(1,n);
parfor j = 1:n
    vj = V(:,j);
    Uj = U(:,IDX(:,j));
    UUj = Uj*Uj';
    term1 = vj'*UUj*vj;
    Sj = ScaleScore(nonzeros(S(:,j)),r,maxS,minS);
    term2 = 2*vj'*Uj*Sj;
    term3 = sum(Sj.^2);
    loss(j) = (term1-term2+term3)/length(nonzeros(S(:,j)));
end
loss = sum(loss);
obj = loss+alpha*norm(U,'fro')^2+alpha*norm(U,'fro')^2-2*alpha*trace(U*X')-2*beta*trace(V*Y');
end
