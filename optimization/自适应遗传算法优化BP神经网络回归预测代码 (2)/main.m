%% ����Ӧ����ͱ�����ʵ��Ŵ��㷨�Ż�BP������Ԥ�����
%{
1. ������ʣ���������������ӣ�ʹ�ý�������ע�ؽ������㣬ȫ����������ǿ
         ��ʽ��  Pc = 0.3 * tan(iteration/Max_iteration * pi/2) + 0.1


2. ������ʣ���������������ӣ�ʹ�ý�������ע�ر������㣬�ֲ���������ǿ
        ��ʽ��  Pc = 0.05 * sin(iteration/Max_iteration * pi/2) + 0.02

%}

%% ��ʼ��
clear
close all
clc
warning off

%% ���ݶ�ȡ
data=xlsread('����.xlsx','Sheet1','A1:N252'); %%ʹ��xlsread������ȡEXCEL�ж�Ӧ��Χ�����ݼ���

%�����������
input=data(:,1:end-1);    %data�ĵ�һ��-�����ڶ���Ϊ����ָ��
output=data(:,end);  %data�������һ��Ϊ�����ָ��ֵ

N=length(output);   %ȫ��������Ŀ
testNum=15;   %�趨����������Ŀ
trainNum=N-testNum;    %����ѵ��������Ŀ

%% ����ѵ���������Լ�
input_train = input(1:trainNum,:)';
output_train =output(1:trainNum)';
input_test =input(trainNum+1:trainNum+testNum,:)';
output_test =output(trainNum+1:trainNum+testNum)';

%% ���ݹ�һ��
[inputn,inputps]=mapminmax(input_train,0,1);
[outputn,outputps]=mapminmax(output_train);
inputn_test=mapminmax('apply',input_test,inputps);

%% ��ȡ�����ڵ㡢�����ڵ����
inputnum=size(input,2);
outputnum=size(output,2);
disp('/////////////////////////////////')
disp('������ṹ...')
disp(['�����Ľڵ���Ϊ��',num2str(inputnum)])
disp(['�����Ľڵ���Ϊ��',num2str(outputnum)])
disp(' ')
disp('������ڵ��ȷ������...')

%ȷ��������ڵ����
%���þ��鹫ʽhiddennum=sqrt(m+n)+a��mΪ�����ڵ������nΪ�����ڵ������aһ��ȡΪ1-10֮�������
MSE=1e+5; %��ʼ����С���
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
    
    %��������
    net=newff(inputn,outputn,hiddennum);
    % �������
    net.trainParam.epochs=1000;         % ѵ������
    net.trainParam.lr=0.01;                   % ѧϰ����
    net.trainParam.goal=0.000001;        % ѵ��Ŀ����С���
    % ����ѵ��
    net=train(net,inputn,outputn);
    an0=sim(net,inputn);  %������
    mse0=mse(outputn,an0);  %����ľ������
    disp(['������ڵ���Ϊ',num2str(hiddennum),'ʱ��ѵ�����ľ������Ϊ��',num2str(mse0)])
    
    %������ѵ�������ڵ�
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['��ѵ�������ڵ���Ϊ��',num2str(hiddennum_best),'����Ӧ�ľ������Ϊ��',num2str(MSE)])

%% �������������ڵ��BP������
disp(' ')
disp('��׼��BP�����磺')
net0=newff(inputn,outputn,hiddennum_best,{'tansig','purelin'},'trainlm');% ����ģ��

%�����������
net0.trainParam.epochs=1000;         % ѵ����������������Ϊ1000��
net0.trainParam.lr=0.01;                   % ѧϰ���ʣ���������Ϊ0.01
net0.trainParam.goal=0.00001;                    % ѵ��Ŀ����С����������Ϊ0.0001
net0.trainParam.show=25;                % ��ʾƵ�ʣ���������Ϊÿѵ��25����ʾһ��
net0.trainParam.mc=0.01;                 % ��������
net0.trainParam.min_grad=1e-6;       % ��С�����ݶ�
net0.trainParam.max_fail=6;               % ���ʧ�ܴ���

%��ʼѵ��
net0=train(net0,inputn,outputn);

%Ԥ��
an0=sim(net0,inputn_test); %��ѵ���õ�ģ�ͽ��з���

%Ԥ��������һ����������
test_simu0=mapminmax('reverse',an0,outputps); %�ѷ���õ������ݻ�ԭΪԭʼ��������
%���ָ��
[mae0,mse0,rmse0,mape0,error0,errorPercent0]=calc_error(output_test,test_simu0);

%% �Ŵ��㷨�㷨Ѱ����Ȩֵ��ֵ
disp(' ')
disp('GA�Ż�BP�����磺')
net=newff(inputn,outputn,hiddennum_best,{'tansig','purelin'},'trainlm');% ����ģ��

%�����������
net.trainParam.epochs=50;         % ѵ������         ��ѭ�������������СһЩ
net.trainParam.lr=0.01;                   % ѧϰ���ʣ���������Ϊ0.01
net.trainParam.goal=0.00001;                    % ѵ��Ŀ����С����������Ϊ0.0001
net.trainParam.show=25;                % ��ʾƵ�ʣ���������Ϊÿѵ��25����ʾһ��
net.trainParam.mc=0.01;                 % ��������
net.trainParam.min_grad=1e-6;       % ��С�����ݶ�
net.trainParam.max_fail=6;               % ���ʧ�ܴ���

%% �Ŵ��㷨�����Ѳ���
maxgen=50;                          %��������������������
sizepop=30;                         %��Ⱥ��ģ
pcross=0.8;                       %�������ѡ��0��1֮��
pmutation=0.1;                    %�������ѡ��0��1֮��

%�ڵ�����������������Ȩֵ��������ֵ�����������Ȩֵ�������ֵ��4���������һ��Ⱦɫ�壩
numsum=inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+outputnum;%21����10,5,5,1

lenchrom=ones(1,numsum);%���峤�ȣ���ʱ�����ΪȾɫ�峤�ȣ���1��numsum�еľ���      
bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    %��numsum��2�еĴ������󣬵�1����-3����2����3

%------------------------------------------------------��Ⱥ��ʼ��--------------------------------------------------------
individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %����Ⱥ��Ϣ����Ϊһ���ṹ�壺10���������Ӧ��ֵ��10��Ⱦɫ�������Ϣ
avgfitness=[];                      %ÿһ����Ⱥ��ƽ����Ӧ��,һά
bestfitness=[];                     %ÿһ����Ⱥ�������Ӧ��
bestchrom=[];                       %��Ӧ����õ�Ⱦɫ�壬���������Ϣ
%��ʼ����Ⱥ
for i=1:sizepop
    %�������һ����Ⱥ
    individuals.chrom(i,:)=Code(lenchrom,bound);    %���루binary�������ƣ���grey�ı�����Ϊһ��ʵ����float�ı�����Ϊһ��ʵ��������
    x=individuals.chrom(i,:);
    %������Ӧ��
    individuals.fitness(i)=fun(x,inputnum,hiddennum_best,outputnum,net,inputn,outputn,output_train,inputn_test,outputps,output_test);   %Ⱦɫ�����Ӧ��
end


%����õ�Ⱦɫ��
[bestfitness, bestindex]=min(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  %��õ�Ⱦɫ�壬��10����������ѡ����
avgfitness=sum(individuals.fitness)/sizepop; %Ⱦɫ���ƽ����Ӧ��(���и�����Ӧ�Ⱥ� / ������)
% ��¼ÿһ����������õ���Ӧ��
trace=zeros(maxgen,1); 
 
%%���������ѳ�ʼ��ֵ��Ȩֵ
% h=waitbar(0,'GA optimization...');
% ������ʼ
for i=1:maxgen
%     pcross = 0.3 * tan(i/maxgen * pi/2) + 0.1;  % �������
%     pmutation = 0.05 * sin(i/maxgen * pi/2) + 0.02;     % ������� 
    % ѡ��
    individuals=Select(individuals,sizepop); 
    %����
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop);
    % ����
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
    
    % ������Ӧ�� 
    for j=1:sizepop
        x=individuals.chrom(j,:); %������Ϣ
        individuals.fitness(j)=fun(x,inputnum,hiddennum_best,outputnum,net,inputn,outputn,output_train,inputn_test,outputps,output_test);  %����ÿ���������Ӧ��ֵ 
    end
    
    %�ҵ���С�������Ӧ�ȵ�Ⱦɫ�弰��������Ⱥ�е�λ��
    [newbestfitness,newbestindex]=min(individuals.fitness);%�����Ӧ��ֵ
    [worestfitness,worestindex]=max(individuals.fitness);
    % ���Ÿ������
    if bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    individuals.chrom(worestindex,:)=bestchrom;%ȡ�������ģ��൱����̭
    individuals.fitness(worestindex)=bestfitness;
    
    trace(i)=bestfitness; %��¼ÿһ����������õ���Ӧ��
%     waitbar(i/maxgen,h)
end
% close(h)

%% ���ƽ�������
figure
plot(trace,'r-','linewidth',2)
xlabel('��������')
ylabel('�������')
legend('�����Ӧ��')
title('GA�Ľ�����������')
w1=bestchrom(1:inputnum*hiddennum_best);         %����㵽�м���Ȩֵ
B1=bestchrom(inputnum*hiddennum_best+1:inputnum*hiddennum_best+hiddennum_best);   %�м������Ԫ��ֵ
w2=bestchrom(inputnum*hiddennum_best+hiddennum_best+1:inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum);   %�м�㵽������Ȩֵ
B2=bestchrom(inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+1:inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+outputnum);   %��������Ԫ��ֵ
%�����ع�
set_rand();
net.iw{1,1}=reshape(w1,hiddennum_best,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum_best);
net.b{1}=reshape(B1,hiddennum_best,1);
net.b{2}=reshape(B2,outputnum,1);

%% �Ż����������ѵ��
net=train(net,inputn,outputn);%��ʼѵ��������inputn,outputn�ֱ�Ϊ�����������

%% �Ż�������������
an1=sim(net,inputn_test);
test_simu1=mapminmax('reverse',an1,outputps); %�ѷ���õ������ݻ�ԭΪԭʼ��������
%���ָ��
[mae1,mse1,rmse1,mape1,error1,errorPercent1]=calc_error(output_test,test_simu1);

%% ��ͼ
figure
plot(output_test,'b-*','linewidth',1)
hold on
plot(test_simu0,'r-v','linewidth',1,'markerfacecolor','r')
hold on
plot(test_simu1,'k-o','linewidth',1,'markerfacecolor','k')
legend('��ʵֵ','BPԤ��ֵ','GA-BPԤ��ֵ')
xlabel('�����������')
ylabel('ָ��ֵ')
title('GA�Ż�ǰ���BP������Ԥ��ֵ����ʵֵ�Ա�ͼ')

figure
plot(error0,'rv-','markerfacecolor','r')
hold on
plot(error1,'ko-','markerfacecolor','k')
legend('BPԤ�����','GA-BPԤ�����')
xlabel('�����������')
ylabel('Ԥ��ƫ��')
title('GA�Ż�ǰ���BP������Ԥ��ֵ����ʵֵ���Ա�ͼ')

figure
bar(abs([error0; error1]'))
legend('BP', 'GABP')


disp(' ')
disp('/////////////////////////////////')
disp('��ӡ������')
disp('�������     ʵ��ֵ      BPԤ��ֵ  GA-BPֵ   BP���   GA-BP���')
for i=1:testNum
    disp([i output_test(i),test_simu0(i),test_simu1(i),error0(i),error1(i)])
end