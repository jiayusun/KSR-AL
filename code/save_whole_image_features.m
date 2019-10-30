clc;clear;close all;
addpath(genpath('./sds_eccv2014-master'));
addpath(genpath('./gop_1.3'));
test_path='./DUT-train-Image/';
test_listname = textread('./image_list_2000/zhanbii/c3_name.txt','%s');
test_num=length(test_listname);
model_def_file='./sds_eccv2014-master/prototxts/pinetwork_extract_fc7.prototxt';
model_file='./sds_eccv2014-master/sds_pretrained_models/nets/C';
assert(exist(model_def_file, 'file')>0);
assert(exist(model_file, 'file')>0);
rcnn_model=rcnn_create_model(model_def_file,model_file);
rcnn_model=rcnn_load_model(rcnn_model);
caffe.set_mode_gpu();
load('./trained_model/trained_model_duts_749/coeff.mat');
load('./trained_model/trained_model_duts_749/Method.mat');
load('./trained_model/trained_model_duts_749/train.mat');
load('./trained_model/trained_model_duts_749/w.mat');
feature_path='./features/duts/feat3/'; 
mkdir(feature_path);
dim=size(train,2);
tic();
for it=1:test_num
    it
    imgname=test_listname{it};
    img_name=[test_path imgname(1:end-3) 'jpg']; 
    I=imread(img_name); 
    out_name_features = [feature_path 'feat_' imgname(1:end-4) '.txt']
    txt_file_all=fopen(out_name_features,'w')    
    box = [0,0,size(I,2),size(I,1)];
    mask = ones(size(I,1),size(I,2));
    feats=rcnn_features_pi_2(I, mask, box, rcnn_model);
    featts=bsxfun(@minus,feats,mean(feats))*coeff(:,1:dim);
    sam_test=double(featts); 
    [Test_new] = compute_rank2(Method, train, sam_test);
        fprintf(txt_file_all,'%s\n',Test_new);    
     fclose(txt_file_all);
end
