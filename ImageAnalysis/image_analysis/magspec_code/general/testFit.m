clear all
close all

X = [0:0.02:3];
p = [5,-10,-15,-18,-20,-1];
p = [1,-1,-1,-1,-1,-1];
Y = polyval(p,X);

tfct = 3;   % polynomial
argf = 5;   % fifth order
N = 4;
be = 15/N;
ro = 10;
d = round(N/10);
nvar = 6;
span = [-20,20;-20,20;-20,20;-20,20;-20,20;-20,20];
init = [1,-1,-1,-1,-1,1];
tol = 1e-5;
itmax = 2000;
durmax = inf;

[best,cost,nbiter,dur,evolution]=gcs(X,Y,tfct,argf,N,be,ro,d,nvar,span,init',tol,itmax,durmax);

best(6:-1:1)
cost

figure(1)
plot(X,Y,'r-')
hold on
plot(X,polyval(best(6:-1:1),X),'bo')
hold off
