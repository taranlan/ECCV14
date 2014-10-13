function bk_train(conf)

options = conf.options;
name = conf.name;
data_train = conf.data_train;
train_clip = conf.train_clip;

action_n = [5 25 60]; % number of classes in each layer
orient_n = 5;
featdims = 3;

load('../subcategory/mat/data_l1.mat', 'state_idx');
for i = 1:action_n(1)-1 % the 5-th class is "no-action", and doesn't have subcategories
  for j = 1:orient_n
    k = (i-1) * orient_n + j;
    action_fine_n(k) = length(state_idx{i}{j});
  end
end

totaldims = action_n(1) * featdims + action_n(1) * action_n(2) * featdims + ...
            action_n(2) * action_n(3) * featdims;

for i = 1:length(data_train)

  flow_n{i} = data_train(i).flow_n;

  feat_a1{i} = [data_train(i).score_action; data_train(i).score_action_flow];
  feat_a2{i} = [data_train(i).score_action_orient; ...
                data_train(i).score_action_orient_flow];
  feat_a3{i} = [data_train(i).score_action_fine(1:action_n(3)); ...
                data_train(i).score_action_fine_flow(1:action_n(3))];
            
  label{i}(1:2) = data_train(i).label(1:2);  % label: first 3 dimensions: the category labels; 4-th dimension: timestamp
  label{i}(4) = data_train(i).label(3);
  
  action_orient = (label{i}(1) - 1) * orient_n + label{i}(2);
  
  if label{i}(1) == 5 % no-action
    label{i}(3:4) = 0;
    continue;
  end
  
  if action_orient == 1
    idx_start = 1;
  else
    idx_start = sum(action_fine_n(1:action_orient-1)) + 1;
  end
  idx_fine = idx_start : sum(action_fine_n(1:action_orient));
  [val idx] = max(data_train(i).score_action_fine(idx_fine));
  label{i}(3) = idx_start + idx - 1;

end

for i = 1:size(train_clip, 1)
  clip.idx{i} = train_clip(i, :);
end

auxdata = struct('action_n', action_n, 'orient_n', orient_n, 'flow_n', flow_n, ...
                 'action_fine_n', action_fine_n, 'clip', clip, ...
                 'feat_a1', feat_a1, 'feat_a2', feat_a2, 'feat_a3', feat_a3, ...
                 'totaldims', totaldims, 'label', label, 'name', name);

% smart initializations
%load('../l1_clip/mat/weight');
load('mat/weight_ini'); % weight learnt from the baseline with only top 2 layers
w0 = ones(1,totaldims);
w0(1:length(w)) = w;

idx_start = action_n(1) * featdims + action_n(1) * action_n(2) * featdims;
for i = 1:action_n(2)
  % set the weights corresponding to the fine-grained subcateogries of "no-action" to zero
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


