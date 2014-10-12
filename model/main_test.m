function main_test(delta_t, clip_len)

addpath(genpath('../util/'));

load('../subcategory/mat/pretrain_data_clip.mat');
load('mat/weight.mat');

conf.data_test = data_test;
conf.w = w;
conf.delta_t = delta_t;
conf.clip_len = clip_len;

bk_test(conf);
