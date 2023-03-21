%% 自适应交叉和变异概率的遗传算法优化BP神经网络预测代码
%{
1. 交叉概率：随迭代次数逐渐增加，使得进化后期注重交叉运算，全局搜索能力强
         公式：  Pc = 0.3 * tan(iteration/Max_iteration * pi/2) + 0.1


2. 变异概率：随迭代次数逐渐增加，使得进化后期注重变异运算，局部搜索能力强
        公式：  Pc = 0.05 * sin(iteration/Max_iteration * pi/2) + 0.02

%}

%% 初始化
clear
close all
clc
warning off

%% 数据读取
data=xlsread('数据.xlsx','Sheet1','A1:N252'); %%使用xlsread函数读取EXCEL中对应范围的数据即可

%输入输出数据
input=data(:,1:end-1);    %data的第一列-倒数第二列为特征指标
output=data(:,end);  %data的最后面一列为输出的指标值

N=length(output);   %全部样本数目
testNum=15;   %设定测试样本数目
trainNum=N-testNum;    %计算训练样本数目

%% 划分训练集、测试集
input_train = input(1:trainNum,:)';
output_train =output(1:trainNum)';
input_test =input(trainNum+1:trainNum+testNum,:)';
output_test =output(trainNum+1:trainNum+testNum)';

%% 数据归一化
[inputn,inputps]=mapminmax(input_train,0,1);
[outputn,outputps]=mapminmax(output_train);
inputn_test=mapminmax('apply',input_test,inputps);

%% 获取输入层节点、输出层节点个数
inputnum=size(input,2);
outputnum=size(output,2);
disp('/////////////////////////////////')
disp('神经网络结构...')
disp(['输入层的节点数为：',num2str(inputnum)])
disp(['输出层的节点数为：',num2str(outputnum)])
disp(' ')
disp('隐含层节点的确定过程...')

%确定隐含层节点个数
%采用经验公式hiddennum=sqrt(m+n)+a，m为输入层节点个数，n为输出层节点个数，a一般取为1-10之间的整数
MSE=1e+5; %初始化最小误差
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
    
    %构建网络
    net=newff(inputn,outputn,hiddennum);
    % 网络参数
    net.trainParam.epochs=1000;         % 训练次数
    net.trainParam.lr=0.01;                   % 学习速率
    net.trainParam.goal=0.000001;        % 训练目标最小误差
    % 网络训练
    net=train(net,inputn,outputn);
    an0=sim(net,inputn);  %仿真结果
    mse0=mse(outputn,an0);  %仿真的均方误差
    disp(['隐含层节点数为',num2str(hiddennum),'时，训练集的均方误差为：',num2str(mse0)])
    
    %更新最佳的隐含层节点
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['最佳的隐含层节点数为：',num2str(hiddennum_best),'，相应的均方误差为：',num2str(MSE)])

%% 构建最佳隐含层节点的BP神经网络
disp(' ')
disp('标准的BP神经网络：')
net0=newff(inputn,outputn,hiddennum_best,{'tansig','purelin'},'trainlm');% 建立模型

%网络参数配置
net0.trainParam.epochs=1000;         % 训练次数，这里设置为1000次
net0.trainParam.lr=0.01;                   % 学习速率，这里设置为0.01
net0.trainParam.goal=0.00001;                    % 训练目标最小误差，这里设置为0.0001
net0.trainParam.show=25;                % 显示频率，这里设置为每训练25次显示一次
net0.trainParam.mc=0.01;                 % 动量因子
net0.trainParam.min_grad=1e-6;       % 最小性能梯度
net0.trainParam.max_fail=6;               % 最高失败次数

%开始训练
net0=train(net0,inputn,outputn);

%预测
an0=sim(net0,inputn_test); %用训练好的模型进行仿真

%预测结果反归一化与误差计算
test_simu0=mapminmax('reverse',an0,outputps); %把仿真得到的数据还原为原始的数量级
%误差指标
[mae0,mse0,rmse0,mape0,error0,errorPercent0]=calc_error(output_test,test_simu0);

%% 遗传算法算法寻最优权值阈值
disp(' ')
disp('GA优化BP神经网络：')
net=newff(inputn,outputn,hiddennum_best,{'tansig','purelin'},'trainlm');% 建立模型

%网络参数配置
net.trainParam.epochs=50;         % 训练次数         ，循环体迭代，设置小一些
net.trainParam.lr=0.01;                   % 学习速率，这里设置为0.01
net.trainParam.goal=0.00001;                    % 训练目标最小误差，这里设置为0.0001
net.trainParam.show=25;                % 显示频率，这里设置为每训练25次显示一次
net.trainParam.mc=0.01;                 % 动量因子
net.trainParam.min_grad=1e-6;       % 最小性能梯度
net.trainParam.max_fail=6;               % 最高失败次数

%% 遗传算法求解最佳参数
maxgen=50;                          %进化代数，即迭代次数
sizepop=30;                         %种群规模
pcross=0.8;                       %交叉概率选择，0和1之间
pmutation=0.1;                    %变异概率选择，0和1之间

%节点总数：输入隐含层权值、隐含阈值、隐含输出层权值、输出阈值（4个基因组成一条染色体）
numsum=inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+outputnum;%21个，10,5,5,1

lenchrom=ones(1,numsum);%个体长度，暂时先理解为染色体长度，是1行numsum列的矩阵      
bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    %是numsum行2列的串联矩阵，第1列是-3，第2列是3

%------------------------------------------------------种群初始化--------------------------------------------------------
individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %将种群信息定义为一个结构体：10个个体的适应度值，10条染色体编码信息
avgfitness=[];                      %每一代种群的平均适应度,一维
bestfitness=[];                     %每一代种群的最佳适应度
bestchrom=[];                       %适应度最好的染色体，储存基因信息
%初始化种群
for i=1:sizepop
    %随机产生一个种群
    individuals.chrom(i,:)=Code(lenchrom,bound);    %编码（binary（二进制）和grey的编码结果为一个实数，float的编码结果为一个实数向量）
    x=individuals.chrom(i,:);
    %计算适应度
    individuals.fitness(i)=fun(x,inputnum,hiddennum_best,outputnum,net,inputn,outputn,output_train,inputn_test,outputps,output_test);   %染色体的适应度
end


%找最好的染色体
[bestfitness, bestindex]=min(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  %最好的染色体，从10个个体中挑选到的
avgfitness=sum(individuals.fitness)/sizepop; %染色体的平均适应度(所有个体适应度和 / 个体数)
% 记录每一代进化中最好的适应度
trace=zeros(maxgen,1); 
 
%%迭代求解最佳初始阀值和权值
% h=waitbar(0,'GA optimization...');
% 进化开始
for i=1:maxgen
%     pcross = 0.3 * tan(i/maxgen * pi/2) + 0.1;  % 交叉概率
%     pmutation = 0.05 * sin(i/maxgen * pi/2) + 0.02;     % 变异概率 
    % 选择
    individuals=Select(individuals,sizepop); 
    %交叉
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop);
    % 变异
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
    
    % 计算适应度 
    for j=1:sizepop
        x=individuals.chrom(j,:); %个体信息
        individuals.fitness(j)=fun(x,inputnum,hiddennum_best,outputnum,net,inputn,outputn,output_train,inputn_test,outputps,output_test);  %计算每个个体的适应度值 
    end
    
    %找到最小和最大适应度的染色体及它们在种群中的位置
    [newbestfitness,newbestindex]=min(individuals.fitness);%最佳适应度值
    [worestfitness,worestindex]=max(individuals.fitness);
    % 最优个体更新
    if bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    individuals.chrom(worestindex,:)=bestchrom;%取代掉最差的，相当于淘汰
    individuals.fitness(worestindex)=bestfitness;
    
    trace(i)=bestfitness; %记录每一代进化中最好的适应度
%     waitbar(i/maxgen,h)
end
% close(h)

%% 绘制进化曲线
figure
plot(trace,'r-','linewidth',2)
xlabel('进化代数')
ylabel('均方误差')
legend('最佳适应度')
title('GA的进化收敛曲线')
w1=bestchrom(1:inputnum*hiddennum_best);         %输入层到中间层的权值
B1=bestchrom(inputnum*hiddennum_best+1:inputnum*hiddennum_best+hiddennum_best);   %中间各层神经元阈值
w2=bestchrom(inputnum*hiddennum_best+hiddennum_best+1:inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum);   %中间层到输出层的权值
B2=bestchrom(inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+1:inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+outputnum);   %输出层各神经元阈值
%矩阵重构
set_rand();
net.iw{1,1}=reshape(w1,hiddennum_best,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum_best);
net.b{1}=reshape(B1,hiddennum_best,1);
net.b{2}=reshape(B2,outputnum,1);

%% 优化后的神经网络训练
net=train(net,inputn,outputn);%开始训练，其中inputn,outputn分别为输入输出样本

%% 优化后的神经网络测试
an1=sim(net,inputn_test);
test_simu1=mapminmax('reverse',an1,outputps); %把仿真得到的数据还原为原始的数量级
%误差指标
[mae1,mse1,rmse1,mape1,error1,errorPercent1]=calc_error(output_test,test_simu1);

%% 作图
figure
plot(output_test,'b-*','linewidth',1)
hold on
plot(test_simu0,'r-v','linewidth',1,'markerfacecolor','r')
hold on
plot(test_simu1,'k-o','linewidth',1,'markerfacecolor','k')
legend('真实值','BP预测值','GA-BP预测值')
xlabel('测试样本编号')
ylabel('指标值')
title('GA优化前后的BP神经网络预测值和真实值对比图')

figure
plot(error0,'rv-','markerfacecolor','r')
hold on
plot(error1,'ko-','markerfacecolor','k')
legend('BP预测误差','GA-BP预测误差')
xlabel('测试样本编号')
ylabel('预测偏差')
title('GA优化前后的BP神经网络预测值和真实值误差对比图')

figure
bar(abs([error0; error1]'))
legend('BP', 'GABP')


disp(' ')
disp('/////////////////////////////////')
disp('打印结果表格')
disp('样本序号     实测值      BP预测值  GA-BP值   BP误差   GA-BP误差')
for i=1:testNum
    disp([i output_test(i),test_simu0(i),test_simu1(i),error0(i),error1(i)])
end