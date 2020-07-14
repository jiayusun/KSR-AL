%% 
clc;clear;close all;
addpath(genpath('./sds_eccv2014-master'));
addpath(genpath('./gop_1.3'));

train500_listname = textread('./image_list_2000/train_list_duts_2000.txt','%s'); % the names list of training data
other1500_listname = textread('./image_list_2000/none.txt','%s'); % the names list of training data

%train500_listname = textread('./image_list_2000/train_list_init500.txt','%s'); % the names list of training data
%other1500_listname = textread('./image_list_2000/train_list_others1500.txt','%s'); % the names list of training data

train_num=length(train500_listname);
other_num =length(other1500_listname);
train300_file_all=fopen('./image_list_2000/train_list_duts_init500.txt','w');
other1700_file_all=fopen('./image_list_2000/train_list_duts_others1500.txt','w');
p = randperm(2000,500);
[pp,~] = sort(p);
pplength = length(pp);

for mm=1:1:other_num
    name = other1500_listname{mm};   
    fprintf(other1700_file_all,'%s\n',name);
end

for ii = 1:1:train_num
    if ismember(ii,pp)==1
        name = train500_listname{ii};   
        fprintf(train300_file_all,'%s\n',name);
    else
        name = train500_listname{ii};   
        fprintf(other1700_file_all,'%s\n',name);
    end
end