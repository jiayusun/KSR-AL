clc;clear;close all;
addpath(genpath('./sds_eccv2014-master'));
addpath(genpath('./gop_1.3'));
train_path='./DUT-train-Image/';   
imagt_path='./DUT-train-Mask/'; 
proposal_path_all='./proposal/train_duts_866/'; 
train_listname = textread('./image_list_2000/zhanbii/train_list_duts_t3.txt','%s');
co_sum=0;ro_sum=0;sam_pri=[];sam_inf=[];
ini_ro=1;
ros=1;cols=1;
%% Load R-CNN network
model_def_file='./sds_eccv2014-master/prototxts/pinetwork_extract_fc7.prototxt';
model_file='./sds_eccv2014-master/sds_pretrained_models/nets/C';
assert(exist(model_def_file, 'file')>0);
assert(exist(model_file, 'file')>0);
caffe.set_mode_gpu();
caffe.set_device(0);
rcnn_model=rcnn_create_model(model_def_file,model_file);
rcnn_model=rcnn_load_model(rcnn_model);

if ~exist(proposal_path_all);
    mkdir(proposal_path_all);
end

for it=1:length(train_listname)
    it
    imgname=train_listname{it}; 
    proposal_path = [proposal_path_all imgname];
    if ~exist(proposal_path);
        mkdir(proposal_path);
    end
        
    img_name=[train_path imgname(1:end-3) 'jpg'];
    imggt_name=[imagt_path imgname(1:end-3) 'png'];
    I=imread(img_name);
    
    [masks]=extract_proposal(I);
    I_gt=im2double(imread(imggt_name));
    I_gt=I_gt(:,:,1);
    num_gtpixel=length(find(I_gt==1));   
    pros_num=size(masks,3);
    clear mask
    clear O_rank
    clear boxes
    clear conf
    for j=1:pros_num       
        pro=masks(:,:,j);
        s=I_gt+pro;
        num_salpixel=length(find(s==2));
        num_propixel=length(find(pro==1));
        beta=0.3;
        conf(j)=(1+beta)*num_salpixel/(beta*num_gtpixel+num_propixel);
    end
    [conf,ind]=sort(conf,'descend');
    masks=masks(:,:,ind);
    pos_ind=find(conf>=0.9); 
    pos_num=length(pos_ind);
    if pos_num<2   
        continue
    else
        neg_ind=find(conf<=0.4);  
        neg_num2=length(neg_ind);
        neg_ind1=find(conf>=0.2);
        neg_ind=[min(neg_ind):max(neg_ind1)];
        neg_num=length(neg_ind);
        if neg_num<pos_num
            continue;
        end
        sam=randperm(neg_num);
        sam_ind=sort(sam(1:pos_num));
        cc=size(conf,2);
        conf=[conf(1:pos_num) conf(cc-neg_num2+sam_ind)];
        mask(:,:,1:pos_num)=masks(:,:,1:pos_num);
        mask(:,:,pos_num+1:pos_num*2)=masks(:,:,cc-neg_num2+sam_ind);
        for iii=1:2*pos_num
            out_name_proposal = [proposal_path '/' imgname(1:end-4) sprintf('_%s',conf(iii))  '.png'];
            imwrite(mask(:,:,iii),out_name_proposal);
        end
        
    end

end

