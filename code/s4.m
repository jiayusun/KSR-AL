clc;clear;close all;
addpath(genpath('./sds_eccv2014-master'));
addpath(genpath('./gop_1.3'));
train_path='./MSRA-5000/';    
imagt_path='./MSRA-5000-gt/';
result_path='./trained_model/trained_model_lambda_mu_00002/';
proposal_path_all='./proposal/train_lambda_mu_0000/';
train_listname = textread('./image_list/final/train_list_t3.txt','%s'); 
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

if ~exist(result_path);
    mkdir(result_path);
end

if ~exist('./proposal');
    mkdir('./proposal');
end

for it=1:length(train_listname)
    it
    imgname=train_listname{it};   
    proposal_path = [proposal_path_all imgname];
    proposal_dir=dir([proposal_path,'/*png']);
    pros_num=length(proposal_dir);
    if pros_num == 0
        continue;
    end
    
    img_name=[train_path imgname(1:end-3) 'jpg'];
    imggt_name=[imagt_path imgname(1:end-3) 'png'];
    I=imread(img_name);
    I_gt=im2double(imread(imggt_name));
    I_gt=I_gt(:,:,1);
    num_gtpixel=length(find(I_gt==1));   
    clear mask
    clear O_rank
    clear boxes
    clear conf
    clear masks
   
    for j=1:pros_num
        j
        pro_name = [proposal_path '/' proposal_dir(j).name];
        pro=double(imread(pro_name))/255;
        masks(:,:,j)=pro;
        s=I_gt+pro;
        num_salpixel=length(find(s==2));
        num_propixel=length(find(pro==1));
        beta=0.3;
        conf(j)=(1+beta)*num_salpixel/(beta*num_gtpixel+num_propixel);
    end
    [conf,ind]=sort(conf,'descend');
    masks=masks(:,:,ind);
    pos_ind=find(conf>=0.5);
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
        neg_ind=find(conf<=0.4);
        neg_num=length(neg_ind);       
        k=1;
        for j=1:pos_num
            for w=1:neg_num     
                O_rank(k,pos_ind(j))=1;
                O_rank(k,neg_ind(w))=-1;
                k=k+1;
            end
        end
        pro_num=size(mask,3);
        for j=1:pro_num
            [ross,colss]=find(mask(:,:,j)==1);                    
             boxes(j,:)=[min(colss(:)),min(ross(:)),max(colss(:)),max(ross(:))];
        end
        mask=double(mask);                                                                         
        feat=rcnn_features_pi_2(I, mask, boxes, rcnn_model);      
        
        pro_num=size(feat,1);        
        sam_pri=[sam_pri;feat(1:pro_num/2,:)];
        sam_inf=[sam_inf;feat(pro_num/2+1:end,:)];
        [ro,co]=size(O_rank); 
        co_sum=co_sum+co/2;
        ro_sum=ro_sum+ro;
        po(ros:ro_sum,cols:co_sum)=O_rank(:,1:co/2);
        ne(ros:ro_sum,cols:co_sum)=O_rank(:,co/2+1:end);
        for k=cols:co_sum
            for w=k+1:co_sum
                ix_pair_po_pri(ini_ro,1)=k;
                ix_pair_po_pri(ini_ro,2)=w;
                ini_ro=ini_ro+1;             
            end
        end
        ros=ros+ro;
        cols=cols+co/2;          
    end
end
%% set parameters 
option.beta=3;
option.d=120;
option.epsilon=1.0000e-4;
option.lambda=0;
option.kernel='gaus-rbf';
O_pair=sparse(ro_sum,2*co_sum);
ix_pair_po_inf=ix_pair_po_pri+co_sum;
O_pair=[po ne];
co_sum=size(O_pair,2);

%% construct positive sample pairs
ix_pair_po=[ix_pair_po_pri;ix_pair_po_inf];

%% construct negative sample pairs
[r1,c1]=(find(O_pair==1));
for i=1:size(O_pair,1)
    [r2, c2(i,1)]=(find(O_pair(i,:)==-1));
end
ix_pair_ne=[c1 c2];

%% construct sample pairs
ix_pair=[ix_pair_po;ix_pair_ne];

%% construct label matrix
y1=ones(size(ix_pair_po,1),1);
y2=-1*ones(size(ix_pair_ne,1),1);
y=[y1;y2];

C_O=0.0001*ones(size(O_pair,1),1);

%% pca dimension reduction
sam_train=[sam_pri;sam_inf];
[coeff,score,latent] = pca(sam_train);
lat = cumsum(latent)./sum(latent);
dim = find(lat>0.8,1);
parts_train = score(:,1:dim)';
train=double(parts_train');
save([result_path 'coeff.mat'],'coeff');

%% compute the kernel matrix
Method = struct('rbf_sigma',0);
[K, Method] = ComputeKernel(train, option.kernel, Method);
K= K*size(K,1)/trace(K);
KJ= spalloc(size(ix_pair, 1), size(train,1)^2, 1);
chop_num = 2000;  
for ns = 2001:chop_num:size(ix_pair,1)
    chop_mat = spalloc(chop_num,size(train,1)^2,1);
    n = 1;
    for i = ns:min(chop_num+ns-1,size(ix_pair,1))
        ix_row1= sub2ind(size(K), 1:size(train,1), ones(1,size(train,1))*ix_pair(i,1));
        chop_mat(n,ix_row1) = K(:,ix_pair(i,1))- K(:,ix_pair(i,2));
        ix_row2= sub2ind(size(K), 1:size(train,1), ones(1,size(train,1))*ix_pair(i,2));
        chop_mat(n,ix_row2) = -chop_mat(n,ix_row1);   
        n = n+1;
    end
    if chop_num+ns-1 < size(ix_pair,1)
        KJ(ns:chop_num+ns-1,:) = sparse(chop_mat);
    else 
        KJ(ns:size(ix_pair,1),:) = sparse(chop_mat(1:mod(size(ix_pair,1),chop_num),:));
    end
end
[Method]=PCCA(train,ix_pair,y,option,K,Method,KJ);
C=max(C_O(:));
iter=0;
l_old=0;
w = zeros(option.d,1);
A=Method.P;
[K_test] = compute_rank2_new(Method, train, train);
Method.l_met=0;
while(1) 
    iter=iter+1
    X_new=A*K_test;
    X_new=double(X_new);    
    w = ranksvm(X_new',O_pair,C_O,w,Method.l_met);  
    [Method]=PCCA_new(train,ix_pair,y,option,w,ix_pair_ne,C,Method,K,KJ);
    if abs(Method.l_new-l_old)<option.epsilon && norm(A-Method.P, 'fro')/norm(A, 'fro')<option.epsilon
        break;
    end
    A=Method.P;
    l_old=Method.l_new;
end
save([result_path 'w.mat'],'w');
save([result_path 'train.mat'],'train');
save([result_path 'Method.mat'],'Method');
caffe.reset_all();
