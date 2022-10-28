% 该程序采用的是proximal block coordinate minimization算法，子问题采用ALH算法，指标更新采用全更新技巧
% clear;


% load 'F:\users\wang\matlab\optimization\ML\SVM\data\2_class\a8a.mat'
% X=data{1};Y=data{2};Y(Y~=1)=-1;
% X=full(X);
% Ins=9*35000;X=data{1}(:,1:Ins);Y=data{2}(1:Ins);Y(Y~=1)=-1;
% for i=1:size(X,1)
%     mm=max(abs(X(i,:)));
%     if mm>0
%         X(i,:)=X(i,:)/mm;
%     end
% end
% load 'F:\users\wang\matlab\optimization\ML\SVM\data\2_class\w7a.t.mat'
% Xt=data{1};Yt=data{2};


%%  ------参数设置------------

para.ker='Lin';
para.IterMax=1000000;
para.C=1;
para.CK=100;
para.g=0.1;para.d=2;para.c=0;
para.sigma=0.1;
para.tolfun=1e-4;
para.Fmin_permit_ratio=0.01;
%%
para.Fmin_permit_ratio1=0.001;
para.tol=1e-4;
para.tolfactor_res=10;
para.tolfactor_x=10;
para.Fmin_permit=6;
para.delta=0.5e-4;
para.outiter=100;
if strcmp(para.ker,'Lin')==1 
    para.speed=2;    %要快一点可改为3；
    para.tolfactor_res=100; %%越大越快 %%调节精度参数；越大越快；
    para.Fmin_permit=4;
else    
    para.speed=3;  
end
para.bet=1e1;
para.Smax=5;
para.alh_maxiter=20;
para.alhtol=1e-4;
para.apgtol=1e-4;
para.trun=2;para.trun_factor=1.03;   %linear kernel 10; Gaussian kerbel 5;
para.output=1;
para.value_tol2=min(1e-5,1/length(Y));
%%
tic;[alpha]=SQP_ALH(X,Y,para);toc;
% tic;[alpha]=Dual_L1(X,Y,para);toc;
% tic;[alpha1]=SQP_ALHmodify(X,Y,para);toc;
% tic;[alpha2]=SQP_ALHmodify(X,Y,para);toc;
% tic;[alpha1]=SQP_ALH1(X,Y,para);toc;
%% Testing
Ins=size(X,2);
Y_=predict_fun(X,X,Y,para,alpha);
Errtrain=sum((Y_.*Y)<=0);
fprintf('样本准确率: ')
fprintf([num2str(Ins-Errtrain),'/',num2str(Ins),'=%2.4f%%\n'],(1-Errtrain/Ins)*100)
% Ins=size(X,2);
% Y_=predict_fun(X,X,Y,para,alpha1);
% Errtrain=sum((Y_.*Y)<=0);
% fprintf('样本准确率: ')
% fprintf([num2str(Ins-Errtrain),'/',num2str(Ins),'=%2.4f%%\n'],(1-Errtrain/Ins)*100)
% Ins=size(X,2);
% Y_=predict_fun(X,X,Y,para,alpha2);
% Errtrain=sum((Y_.*Y)<=0);
% fprintf('样本准确率: ')
% fprintf([num2str(Ins-Errtrain),'/',num2str(Ins),'=%2.4f%%\n'],(1-Errtrain/Ins)*100)
% Ins=size(X,2);
% Y_=predict_fun(X,X,Y,para,alpha3);
% Errtrain=sum((Y_.*Y)<=0);
% fprintf('样本准确率: ')
% fprintf([num2str(Ins-Errtrain),'/',num2str(Ins),'=%2.4f%%\n'],(1-Errtrain/Ins)*100)

