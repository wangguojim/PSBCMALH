function K=KernelGauss(X1,u,sigma,X2,v)
% Gaussian Ker
% X1=X(1:4,:);X2=X(5:12,:);
if nargin>3
    [~,Ins1]=size(X1);[~,Ins2]=size(X2);    
    if Ins1==0||Ins2==0
        K=zeros(Ins1,Ins2);
    else
        U1=X1'*X2;         
        [U4,U5]=meshgrid(v,u);
        K=U4+U5-2*U1;
    end
else
    U= X1'*X1;
    [U4,U5]=meshgrid(u,u);
    K=U4+U5-2*U;
end
K=exp(K*(-sigma));
end