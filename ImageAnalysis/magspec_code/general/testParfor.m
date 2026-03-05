% test parallel



j=1000; loop = 2000;

testM = zeros(j,j)+1;

tic
for i=1:loop
    A(i) = sum(sum(i*testM+i));
end
t1=toc;

tic
parfor i=1:loop
    B(i) = sum(sum(i*testM+i));
end
t2=toc;

disp(t1/t2)

%%
 tic
 for ii=1:100
     x(ii) = max( eig( abs( rand( 200 ) ) ) );
 end
 toc

 %%
 matlabpool local 4
 pctRunOnAll system_dependent( 'getpid' )
 tic
 parfor ii=1:100
     x(ii) = max( eig( abs( rand( 200 ) ) ) );
 end
 toc
 matlabpool close
