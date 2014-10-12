function bk_train(conf)

options = conf.options;
name = conf.name;
data_train = conf.data_train;
train_clip = conf.train_clip;

action_n = [5 25 60];
orient_n = 5;
featdims = 3;
%action_fine_n = [2 2 3 2 1 2 3 4 2 1 3 3 1 2 2 3 3 1 2 2];
% load('../subcategory/mat/model/ap_cluster.mat', 'pos_cluster_idx');
% for i = 1:action_n(1)-1
%   for j = 1:orient_n
%       
%     k = (i-1) * orient_n + j;
%     action_fine_n(k) = length(pos_cluster_idx{i}{j});
%     
%   end
% end

load('../subcategory/mat/pos_neg/data_l1.mat', 'state_idx');
for i = 1:action_n(1)-1
  for j = 1:orient_n
      
    k = (i-1) * orient_n + j;
    %action_fine_n(k) = length(pos_cluster_idx{i}{j});
    action_fine_n(k) = length(state_idx{i}{j});
  end
end

%featdims_a = [5 25]; % actions
%featdims_o = 5; % orientation
%R = 6;

totaldims = action_n(1) * featdims + action_n(1) * action_n(2) * featdims + ...
            action_n(2) * action_n(3) * featdims;

% feat_a1 = {data_train.score_action}; % detailed action classes
% feat_a2 = {data_train.score_action_orient}; % detailed action + orientation
% label = {data_train.label}; 

for i = 1:length(data_train)
  %feat_a1{i} = data_train(i).score_action;
%   feat_a2{i} = data_train(i).score_action_orient;
%   feat_a3{i} = data_train(i).score_action_fine;

  flow_n{i} = data_train(i).flow_n;

  feat_a1{i} = [data_train(i).score_action; data_train(i).score_action_flow];
  %feat_a1{i} = [data_train(i).score_action_flow; zeros(1, action_n(1))];
  feat_a2{i} = [data_train(i).score_action_orient; ...
                data_train(i).score_action_orient_flow];
  feat_a3{i} = [data_train(i).score_action_fine(1:action_n(3)); ...
                data_train(i).score_action_fine_flow(1:action_n(3))];
            
  label{i}(1:2) = data_train(i).label(1:2);
  
  action_orient = (label{i}(1) - 1) * orient_n + label{i}(2);
  
  if label{i}(1) == 5
    label{i}(3) = 0;
    continue;
  end
  
  if action_orient == 1
    idx_start = 1;
  else
    idx_start = sum(action_fine_n(1:action_orient-1)) + 1;
  end
  idx_fine = idx_start : sum(action_fine_n(1:action_orient));
  %[val idx] = max(feat_a3{i}(idx_fine));
  [val idx] = max(data_train(i).score_action_fine(idx_fine));
  label{i}(3) = idx_start + idx - 1;

end

for i = 1:size(train_clip, 1)
  clip.idx{i} = train_clip(i, :);
end

% action_r = zeros(1, action_n(1));
% for i = 1:length(label)
% 
%   y = label{i}(1);
%   action_r(y) = action_r(y) + 1;
% 
% end
% 
% action_r = 1 ./ (action_r / sum(action_r));
% action_r = action_r / 10;
% action_r(5) = action_r(5)/1000;

auxdata = struct('action_n', action_n, 'orient_n', orient_n, 'flow_n', flow_n, ...
                 'action_fine_n', action_fine_n, 'clip', clip, ...
                 'feat_a1', feat_a1, 'feat_a2', feat_a2, 'feat_a3', feat_a3, ...
                 'totaldims', totaldims, 'label', label, 'name', name);

% smart initializations
load('../l1_clip/mat/weight');
w0 = ones(1,totaldims);
%w0 = zeros(1,totaldims);
w0(1:length(w)) = w;

% for i = 1:action_n(1)
%   for j = 1:action_n(2)
%       
%     if floor((j - 1) / orient_n) + 1 ~= i
%       idx = action_n(1) * featdims + (i - 1) * action_n(2) * featdims ...
%             + (j - 1) * featdims + 1;
%       w0(idx:idx+featdims-1) = 0;
%     end
%     
%   end
% end

idx_start = action_n(1) * featdims + action_n(1) * action_n(2) * featdims;
%idx_start = action_n(1) * featdims;
%w0(idx_start : end) = 0;
for i = 1:action_n(2)
  
  if i > 20
    idx = idx_start + (20-1) * action_n(3) * featdims;
    w0(idx+1:end) = 0;
    break;
  end    
    
  for j = 1:action_n(3)
      
    if i == 1
      y_start = 1;
    else
      y_start = sum(action_fine_n(1:i-1)) + 1;
    end
    y_end = sum(action_fine_n(1:i));
    
    if j < y_start || j > y_end
      idx = idx_start + (i -1) * action_n(3) * featdims + ... 
            (j - 1) * featdims + 1;  
      w0(idx:idx+featdims-1) = 0;
      continue;
    end
    
  end
  
end

lambda = options.lambda;

[w fbest numfeval fstart] = NRBM(w0,lambda,options, ...
                                    @train_compute, auxdata);
save('mat/weight.mat', 'w');


