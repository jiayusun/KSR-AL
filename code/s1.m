%% test
clc;clear;close all;
addpath(genpath('./sds_eccv2014-master'));
addpath(genpath('./gop_1.3'));
test_path='./DUT-train-Image/';
fusion_path='./saliency_map/duts_749/';
mkdir(fusion_path);
test_listname = textread('./image_list_2000/ratiozi/train_list_duts_other_t2.txt','%s');
test_num=length(test_listname);

%% Load network
model_def_file='./sds_eccv2014-master/prototxts/pinetwork_extract_fc7.prototxt';
model_file='./sds_eccv2014-master/sds_pretrained_models/nets/C';
assert(exist(model_def_file, 'file')>0);
assert(exist(model_file, 'file')>0);
rcnn_model=rcnn_create_model(model_def_file,model_file);
rcnn_model=rcnn_load_model(rcnn_model);
caffe.set_mode_gpu();
caffe.set_device(0);

%% load trained model   ?
load('./trained_model/trained_model_duts_749/coeff.mat');%%$$
load('./trained_model/trained_model_duts_749/Method.mat');%%$$
load('./trained_model/trained_model_duts_749/train.mat');%%$$
load('./trained_model/trained_model_duts_749/w.mat');%%$$
dim=size(train,2);
ratioz_all = [];
tic();
for it=1:test_num
    it
    imgname=test_listname{it};
    img_name=[test_path imgname(1:end-3) 'jpg']; 
    fusion_name=[fusion_path imgname(1:end-4) '.png'];
    I=imread(img_name); 
   if ~exist(fusion_name)
        %% extract proposals and preprocess proposals
        [masks]=extract_proposal(I);
        pros_num=size(masks,3);
        %% prepare data for feature extraction
        for j=1:pros_num
            [ros,cols]=find(masks(:,:,j)==1);
             boxes(j,:)=[min(cols(:)),min(ros(:)),max(cols(:)),max(ros(:))];
        end
        masks=double(masks);
        feats=rcnn_features_pi_2(I, masks, boxes, rcnn_model);
        %% pca dimension reduction
        feats=bsxfun(@minus,feats,mean(feats))*coeff(:,1:dim);
        sam_test=double(feats);
        %% computer the kernel matrix
        [Test_new] = compute_rank2(Method, train, sam_test);
        %% compute ranking scores 
        rank_val=Test_new'*w; 
        %% final saliency map: a weighted fusion of top-16 ranked masks;
        mask_num=size(masks,3);
        [rank_val,inde]=sort(rank_val,'descend');
        masks=masks(:,:,inde);
        rank=rank_val(1:16);
        [rows,cols]=size(masks(:,:,1));
        result=zeros(rows,cols);
        for j=1:16
            result=result+masks(:,:,j)*exp(2*rank(j));
        end
        %% change by ai
        rank_val = (rank_val-min(rank_val))/(max(rank_val)-min(rank_val));   
        ind_mid = (rank_val<0.9).*(rank_val>0.4);
        ratioz = sum(ind_mid)/length(rank_val);
        ratioz_all = [ratioz_all;ratioz];
        %% normalization
        index=find(result~=0);
        mi=result(index);
        result=(result-min(mi(:)))/(max(mi(:))-min(mi(:)));
        ind=find(result==(-min(mi(:)))/(max(mi(:))-min(mi(:))));
        result(ind)=0;
        imwrite(result,fusion_name,'png');
        clear boxes        
   end 
end

save ./image_list_2000/ratioz-duts_749.txt -ascii ratioz_all; %$$$
[ratioz_all,~] = sort(ratioz_all,'descend');
toc();
