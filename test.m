%generate a toy random user-item matrix of 100 users and 50 items.
S = rand(100,50);
S(S>0.5) = 0;

%max rating is 5 and minimum rating is 1
S = S*5;
S = ceil(S);

S = sparse(S);
ST = S';
IDX = (S~=0);
IDXT = IDX';

%apply initialization
option.Init = true;

%target bit size is 8
r = 8;

alpha =0.01;

beta = 0.01;

option.debug = true;

%number of iterations
option.maxItr = 20;

[B,D,X,Y] = DCF(5,1,S, ST, IDX, IDXT, r, alpha, beta, option);