function main_train

addpath(genpath('../util/'));

load('../subcategory/mat/pretrain_data_clip.mat', 'data_train', 'train_clip');
  
conf.data_train = data_train;
conf.train_clip = train_clip;
conf.name.path = 'unary_clip';
conf.options = struct('lambda', 1,... % regularization parameter
                  'maxiter',60,... % max number of iteration
 	             'maxCP',100,... % max number of cutting plane
         	     'EPS',0.001,... % stop criteria gap=3%
                  'fpositive',  1, ... 
                  'nonconvex' , 0);

bk_train(conf);
