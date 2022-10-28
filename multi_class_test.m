% 该程序采用的是proximal block coordinate minimization算法，子问题采用ALH算法，指标更新采用全更新技巧
% 多分类问题：one vs all
% clear;
%%   load database

load('F:\users\wang\matlab\optimization\ML\SVM\data\multi_class\mnist8m_3.mat')
X=data{1};Ylabel=data{2};
% Ins=20000;X=data{1}(:,1:Ins);Ylabel=data{2}(1:Ins);

% X=X*255;
% X=full(X);
% for i=1:size(X,1)
%     mm=max(abs(X(i,:)));
%     if mm>0
%         X(i,:)=X(i,:)/mm;
%     end
% end

% Ylabel=1011*Ylabel;
%%  ------参数设置------------
para.ker='Lin';
para.IterMax=1000000;
para.C=10;
para.CK=100;
para.g=1;para.d=2;para.c=0;
para.sigma=0.1;
para.tolfun=1e-5;
para.Fmin_permit_ratio=0.001;
para.Lambdacho=0;
%%
para.Fmin_permit_ratio1=0.0001;
para.tol=1e-4;
para.apgtol=1e-4;
para.tolfactor_res=10;
para.tolfactor_x=10;
para.Fmin_permit=6;
para.delta=0.5e-4;
if strcmp(para.ker,'Lin')==1    
    para.speed=2;
    para.tolfactor_res=100;
    para.Fmin_permit=4;
else    
    para.speed=3;
end
para.bet=1e1;
para.Smax=5;
para.alh_maxiter=20;
para.alhtol=1e-4;
para.trun=2;para.trun_factor=1.03;   %linear kernel 10; Gaussian kerbel 5;
para.output=1;
para.outiter=100;
para.value_tol2=min(1e-5,10/length(Ylabel));
%%  ======================One Vs All===================== %%
% % % % % %%%------------Training-----------------
fprintf('===============Multi classification: One Vs all================\n')
Ins=length(Ylabel);
Label=unique(Ylabel);
Label_K=length(Label);
ALPHA=zeros(Ins,Label_K);
YAll=zeros(Ins,Label_K);
YT=YAll;
total_time=0;
ERR_K=[];TIME=[];ACC=[];
for kind=1:Label_K
    Y=zeros(Ins,1);
    Y(Ylabel==Label(kind))=1;
    Y(Ylabel~=Label(kind))=-1;
    YAll(:,kind)=Y;
    tic;
%     tim=tic;[alpha]=SQP_ALH(X,Y,para);tim=toc(tim);
        tim=tic;[alpha]=Dual_L1(X,Y,para);tim=toc(tim);    
    toc;
    YT(:,kind)=predict_fun(X,X,Y,para,alpha);
    Errtrain=sum(( YT(:,kind).*Y)<0);
    fprintf('Accuracy: ')
    fprintf([num2str(Ins-Errtrain),'/',num2str(Ins),'=%2.4f%%\n'],(1-Errtrain/Ins)*100)
    if (1-Errtrain/Ins)<0.5
        ERR_K(end+1)=kind;
    end
    
    total_time=total_time+tim;
    ALPHA(:,kind)=alpha;
    TIME(end+1)=tim;
    ACC(end+1)=1-Errtrain/Ins;
end
fprintf(['Running time','=%2.4f\n'],total_time)
% % %-------- Testing-------------------
Inst=size(X,2);
[~,Yt_]=max(YT');
Yt_=Label(Yt_');
Errtrain=sum((Yt_-Ylabel)~=0);
fprintf('Accuracy: ')
fprintf([num2str(Ins-Errtrain),'/',num2str(Inst),'=%2.4f%%\n'],(1-Errtrain/Inst)*100)

%% ===============One Vs One===========================
% ------------Trainging-----------------
% fprintf('===============Multi classification: One Vs One================\n')
% Ins=length(Ylabel);
% Label=unique(Ylabel);
% Label_K=length(Label);
% ALPHA=cell(Label_K,Label_K);
% YAll=cell(Label_K,Label_K);
% JR=cell(Label_K,Label_K);
% total_time1=0;
% for kind1=1:Label_K-1
%     for kind2=kind1+1:Label_K
%         J=find(Ylabel==Label(kind1)|Ylabel==Label(kind2));
%         JR{kind1,kind2}=J;
%         Y=Ylabel(J);
%         Y(Y==Label(kind1))=1;
%         Y(Y==Label(kind2))=-1;
%         YAll{kind1,kind2}=Y;
%         tim=tic;
%         tic;[alpha]=SQP_ALH(X(:,J),Y,para);toc;
%         tim=toc(tim);
%         total_time1=total_time1+tim;
%         ALPHA{kind1,kind2}=alpha;
%     end
% end
% fprintf(['Running time','=%2.4f\n'],total_time1)
% % % -------------------Testing-----------------------%
% Inst=length(Ylabel);
% YT=cell(Label_K,Label_K);
% for kind1=1:Label_K-1
%     for kind2=kind1+1:Label_K
%         YT{kind1,kind2}=predict_fun(X(:,JR{kind1,kind2}),X,YAll{kind1,kind2},para,ALPHA{kind1,kind2});
%         YT{kind2,kind1}=-YT{kind1,kind2};
%     end
% end
% B=zeros(Label_K,Ins);
% for i=1:Label_K
%     A=[];
%     for j=1:Label_K
%         if i~=j;
%        A(:,end+1)=YT{i,j};
%         end
%     end
%     B(i,:)=sum(A'>0);
% end
% [~,Yt_]=max(B);
% Yt_=Label(Yt_');
% Errtrain=sum((Yt_-Ylabel)~=0);
% fprintf('Accuracy: ')
% fprintf([num2str(Ins-Errtrain),'/',num2str(Inst),'=%2.4f%%\n'],(1-Errtrain/Inst)*100)
