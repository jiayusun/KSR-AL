%% 
clc;clear;close all;
addpath(genpath('./sds_eccv2014-master'));
addpath(genpath('./gop_1.3'));


train_listname = textread('./image_list_2000/train_list_init200.txt','%s'); % the names list of training data
train_num=length(train_listname);

txt_file_all=fopen('./image_list_2000/zhanbi/train_list_init200add1.txt','w');
for ii = 1:1:train_num
    name = train_listname{ii};   
    fprintf(txt_file_all,'%s\n',name);
end



test_path='./MSRA-5000/';
test_listname = textread('./image_list_2000/train_list_other1800.txt','%s');


zhanbi_listname = textread('./image_list_2000/zhanbi-rev1_init200.txt','%s');
zhanbi_img_list_order = str2double(zhanbi_listname);
[zhanbi_descent,~] = sort(zhanbi_img_list_order,'descend');
average_val = mean(zhanbi_descent);
standard_dev = std(zhanbi_descent);
lambdaa = 1.145;
zhanbi_critical_value = average_val + (lambdaa * standard_dev);
zhanbi_num=length(zhanbi_img_list_order);
j = 1;
for i = 1:1:zhanbi_num

    if zhanbi_img_list_order(i)>= zhanbi_critical_value
       big_zhanbi_order_numbers(j,1)= i;
       j = j+1;
    end
end

big_zhanbi_quantity = length(big_zhanbi_order_numbers);

for iii = 1:1:big_zhanbi_quantity
    order_number = big_zhanbi_order_numbers(iii);
    img_name = test_listname{order_number};   
    fprintf(txt_file_all,'%s\n',img_name);
    test_listname{order_number}=0;
end

other_txt_file_all=fopen('./image_list_2000/zhanbi/train_list_other1800m1.txt','w');
for iiii= 1:1:size(test_listname)
    if test_listname{iiii}~=0 
       other_img_name = test_listname{iiii};   
       fprintf(other_txt_file_all,'%s\n',other_img_name);
    end
end
