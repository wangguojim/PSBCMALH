% �ó�����õ���proximal block coordinate minimization�㷨�����������ALH�㷨��ָ����²���ȫ���¼���
% clear;

% �ó�����õ���proximal block coordinate minimization�㷨�����������ALH�㷨��ָ����²���ȫ���¼���
% clear;

%% nonlinear kernel

%  load 'F:\users\wang\matlab\optimization\ML\SVM\data\2_class\url_combined.mat'
% 
% Ins=min(20000000,length(data{2}));X=data{1}(:,1:Ins);Y=data{2}(1:Ins);Y(Y~=1)=-1;
% % X=full(X);
% % for i=1:size(X,1)
% %     mm=max(abs(X(i,:)));
% %     if mm>0
% %         X(i,:)=X(i,:)/mm;
% %     end
% % end
% % load 'F:\users\wang\matlab\optimization\ML\SVM\data\2_class\w7a.t.mat'
% % Xt=data{1};Yt=data{2};
% 
% 
% %%  ------��������------------
% datetime
% para.ker='Poly';
% para.IterMax=10500000;
% para.C=10;
% para.CK=200;
% para.g=0.1;para.d=2;para.c=0;
% para.sigma=0.1;
% para.tolfun=1e-4; %%��1e-4��
% para.tolfun2=min(para.tolfun/10,1/length(Y));
% para.Fmin_permit_ratio=0.01;
% para.Fmin_permit_ratio1=0.001;
% %%
% para.tol=1e-4;
% para.tolfactor_res=100; %% %%����tau�Ĵ�С ��100��������Ϊ�ϰ����ȡֵ
% para.tolfactor_x=10; % (10)
% para.tolfactormin=10; %(10)
% para.delta=0.5e-4;
% 
% if strcmp(para.ker,'Lin')==1 
%     para.speed=2;    %Ҫ��һ��ɸ�Ϊ3�� %(2)
%     para.speedmax=3;  %% speed_max Խ�����Խ�� (2)
%     para.Fmin_permit=2;  %(4)(2) %��Ҫ�����
% else   
%     para.tolfactor_res=10; %% %%����tau�Ĵ�С ��100��������Ϊ�ϰ����ȡֵ
%     para.speed=3;  
%     para.speedmax=4;  
%     para.Fmin_permit=2;%(6)
% end
% para.bet=1e1;
% para.Smax=5;
% para.alh_maxiter=20;
% para.alhtol=1e-4;
% para.apgtol=1e-4;
% para.trun=2;para.trun_factor=1.03;   %linear kernel 10; Gaussian kerbel 5;
% para.output=1;
% para.outiter=10;
% %%
% tic;[alpha]=SQP_ALH(X,Y,para);toc;
% % tic;[alpha]=Dual_L1(X,Y,para);toc;
% % tic;[alpha1]=SQP_ALHmodify(X,Y,para);toc;
% % tic;[alpha2]=SQP_ALHmodify(X,Y,para);toc;
% % tic;[alpha1]=SQP_ALH1(X,Y,para);toc;
% datetime
% %% Testing
% % Ins=size(X,2);
% % Y_=predict_fun(X,X,Y,para,alpha);
% % Errtrain=sum((Y_.*Y)<=0);
% % fprintf('����׼ȷ��: ')
% % fprintf([num2str(Ins-Errtrain),'/',num2str(Ins),'=%2.4f%%\n'],(1-Errtrain/Ins)*100)
% % Ins=size(X,2);
% % Y_=predict_fun(X,X,Y,para,alpha1);
% % Errtrain=sum((Y_.*Y)<=0);
% % fprintf('����׼ȷ��: ')
% % fprintf([num2str(Ins-Errtrain),'/',num2str(Ins),'=%2.4f%%\n'],(1-Errtrain/Ins)*100)
% % Ins=size(X,2);
% % Y_=predict_fun(X,X,Y,para,alpha2);
% % Errtrain=sum((Y_.*Y)<=0);
% % fprintf('����׼ȷ��: ')
% % fprintf([num2str(Ins-Errtrain),'/',num2str(Ins),'=%2.4f%%\n'],(1-Errtrain/Ins)*100)
% % Ins=size(X,2);
% % Y_=predict_fun(X,X,Y,para,alpha3);
% % Errtrain=sum((Y_.*Y)<=0);
% % fprintf('����׼ȷ��: ')
% % fprintf([num2str(Ins-Errtrain),'/',num2str(Ins),'=%2.4f%%\n'],(1-Errtrain/Ins)*100)
% 
% fvalue=1/2*(Y.*alpha)'*X'*X*(Y.*alpha)-sum(alpha);


%%  Linear kernel

load 'F:\wang\matlab\optimization\ML\SVM\data\2_class\url_combined.mat'
X=data{1};Y=data{2};Y(Y~=1)=-1;
% 
% Ins=min(10000000,length(data{2}));X=data{1}(:,1:Ins);Y=data{2}(1:Ins);Y(Y~=1)=-1;
% for i=1:size(X,1)
%     mm=max(abs(X(i,:)));
%     if mm>0
%         X(i,:)=X(i,:)/mm;
%     end
% end
% load 'F:\users\wang\matlab\optimization\ML\SVM\data\2_class\w7a.t.mat'
% Xt=data{1};Yt=data{2};


%%  ------��������------------
datetime
para.ker='Lin';
para.IterMax=1000000;
para.C=1;
para.CK=100;
para.g=0.1;para.d=2;para.c=0;
para.sigma=0.1;
para.tolfun=1e-3; %%��1e-4��
para.tolfun2=min(para.tolfun/10,1/length(Y));
para.Fmin_permit_ratio=0.01;%0.01
para.Fmin_permit_ratio1=0.01;%0.001
%%
para.tol=1e-4;
para.tolfactor_res=100; %% %%����tau�Ĵ�С ��100��������Ϊ�ϰ����ȡֵ
para.tolfactor_x=10; % (10)
para.tolfactormin=100; %(100)
para.delta=0.5e-4;

if strcmp(para.ker,'Lin')==1
    para.speed=2;    %Ҫ��һ��ɸ�Ϊ3�� %(2)
    para.speedmax=3;  %% speed_max Խ�����Խ�� (2)
    para.Fmin_permit=1;  %(4)(2) %��Ҫ�����
else
    para.speed=3;
    para.speedmax=4;
    para.Fmin_permit=4;
end
para.bet=1e1;
para.Smax=5;
para.alh_maxiter=20;
para.alhtol=1e-4;
para.apgtol=1e-4;
para.trun=2;para.trun_factor=1.03;   %linear kernel 10; Gaussian kerbel 5;
para.output=1;
para.outiter=100;
%%
% tic;[alpha]=SQP_ALH(X,Y,para);toc;
tic;[alpha]=Dual_L1(X,Y,para);toc;
% tic;[alpha1]=Dual_L2(X,Y,para);toc;
 
datetime
%% Testing
Ins=size(X,2);
Y_=predict_fun(X,X,Y,para,alpha);
Errtrain=sum((Y_.*Y)<=0);
fprintf('����׼ȷ��: ')
fprintf([num2str(Ins-Errtrain),'/',num2str(Ins),'=%2.4f%%\n'],(1-Errtrain/Ins)*100)
% Ins=size(X,2);
% Y_=predict_fun(X,X,Y,para,alpha1);
% Errtrain=sum((Y_.*Y)<=0);
% fprintf('����׼ȷ��: ')
% fprintf([num2str(Ins-Errtrain),'/',num2str(Ins),'=%2.4f%%\n'],(1-Errtrain/Ins)*100)
% Ins=size(X,2);
% Y_=predict_fun(X,X,Y,para,alpha2);
% Errtrain=sum((Y_.*Y)<=0);
% fprintf('����׼ȷ��: ')
% fprintf([num2str(Ins-Errtrain),'/',num2str(Ins),'=%2.4f%%\n'],(1-Errtrain/Ins)*100)
% Ins=size(X,2);
% Y_=predict_fun(X,X,Y,para,alpha3);
% Errtrain=sum((Y_.*Y)<=0);
% fprintf('����׼ȷ��: ')
% fprintf([num2str(Ins-Errtrain),'/',num2str(Ins),'=%2.4f%%\n'],(1-Errtrain/Ins)*100)

fvalue=1/2*(Y.*alpha)'*X'*X*(Y.*alpha)-sum(alpha)