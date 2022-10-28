% n=20;
% F=ones(n,1);
% for i=1:n
%     F(i)=F(i)+floor(5*rand);
% end
function FA=Sample_choose(F)


Fc=sort(unique(F));
len_Fc=length(Fc);
FA=[];
for i=1:len_Fc   
    LE=find(F==Fc(len_Fc-i+1));
    V=ones(Fc(len_Fc-i+1),1);
    FA=[FA;kron(V,LE)];
end
len_FA=length(FA);J=randperm(len_FA);FA=FA(J);