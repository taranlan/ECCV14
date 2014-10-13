function bk_test(conf)

data_test = conf.data_test;
w = conf.w;
delta_t = conf.delta_t;
clip_len = conf.clip_len;

action_n = [5 25 60];
orient_n = 5;

load('../subcategory/mat/data_l1.mat', 'state_idx');
for i = 1:action_n(1)-1
  for j = 1:orient_n
    k = (i-1) * orient_n + j;
    action_fine_n(k) = length(state_idx{i}{j});
  end
end

n = 0;
flow_n = [];
for i = 1:length(data_test)

  t = data_test(i).label(3);
  if data_test(i).label(1) ~= 5 && (t < delta_t(1) || t > delta_t(2))
    continue;
  end
  
  clip_i = data_test(i).clip;
  if (clip_i(1) > (i-(clip_len-1)/2)) || (clip_i(end) < (i+(clip_len-1)/2))
    continue;
  end
  
  n = n + 1;
  
  flow_n{n} = data_test(i).flow_n;

  label{n} = data_test(i).label;
  clip{n} = [i-(clip_len-1)/2 i+(clip_len-1)/2];
  
  data_o(n).v = data_test(i).v;
  data_o(n).fr = data_test(i).fr;
  data_o(n).bbox = data_test(i).bbox;
  data_o(n).label = data_test(i).label;
  data_o(n).clip = clip{n};
  
end

confmat = zeros(action_n(1));

label_gt = [];
label_pred = [];
for n = 1:length(label)

  disp([int2str(n) ':' int2str(length(label))]);
  
  label_gt = [label_gt; label{n}(1)];
  
  feat_a1_v = [];
  feat_a2_v = [];
  feat_a3_v = [];
  
  feat_a1_tmp = [];
  feat_a2_tmp = [];
  feat_a3_tmp = [];
  for i = clip{n}(1) : clip{n}(2)
    feat_a1_tmp = [feat_a1_tmp; data_test(i).score_action];
    feat_a2_tmp = [feat_a2_tmp; data_test(i).score_action_orient];
    feat_a3_tmp = [feat_a3_tmp; data_test(i).score_action_fine(1:action_n(3))];
  end
  
  feat_a1_v(1,:) = max(feat_a1_tmp, [], 1);
  feat_a2_v(1,:) = max(feat_a2_tmp, [], 1);
  feat_a3_v(1,:) = max(feat_a3_tmp, [], 1);

  feat_a1_tmp = [];
  feat_a2_tmp = [];
  feat_a3_tmp = [];
  flow_n = 0;
  for i = clip{n}(1) : clip{n}(2)
    flow_n = flow_n + data_test(i).flow_n; 
    feat_a1_tmp = [feat_a1_tmp; data_test(i).score_action_flow];
    feat_a2_tmp = [feat_a2_tmp; data_test(i).score_action_orient_flow];
    feat_a3_tmp = [feat_a3_tmp; data_test(i).score_action_fine_flow(1:action_n(3))];
  end
  
  feat_a1_v(2,:) = sum(feat_a1_tmp, 1) / (flow_n+eps);
  feat_a2_v(2,:) = sum(feat_a2_tmp, 1) / (flow_n+eps);
  feat_a3_v(2,:) = sum(feat_a3_tmp, 1) / (flow_n+eps);
  
  label_p = infer_label(w, action_n, orient_n, action_fine_n, ...
                  feat_a1_v, feat_a2_v, feat_a3_v);
  
  label_pred = [label_pred; label_p(1)];
  
  data_o(n).label_pred = label_p;
  
end

for n = 1:size(label_gt, 1)
  confmat(label_gt(n), label_pred(n)) = ...
                        confmat(label_gt(n), label_pred(n)) + 1;
end

confmat

confmat = confmat + eps;
confmat = confmat ./ repmat(sum(confmat, 2), [1 action_n(1)]);
acc = mean(diag(confmat))
acc_all = sum(label_gt == label_pred) / n

confmat

save(['mat/result_' int2str(delta_t(1)) '_' int2str(delta_t(end)) ...
     '_' int2str(clip_len) '.mat'], 'confmat', 'acc', 'acc_all', 'data_o');


function label_pred = infer_label(w, action_n, orient_n, action_fine_n, ...
                                  feat_a1, feat_a2, feat_a3)
    
featdims = 3;
label_n = length(action_n); % n labels for one person
nStates = action_n;
nNodes = length(nStates);
maxState = max(nStates);

adj = zeros(nNodes);
for i = 1:label_n-1
  adj(i,i+1) = 1;
end
adj = adj + adj';

edgeStruct = UGM_makeEdgeStruct(adj, nStates);
nEdges = edgeStruct.nEdges;
nodePot = -inf*ones(nNodes, maxState);
edgePot = -inf*ones(maxState, maxState, nEdges);

for i = 1:action_n(1)
  idx = (i - 1) * featdims + 1;
  nodePot(1,i) = w(idx:idx+featdims-1) * [feat_a1(1,i); feat_a1(2,i); 1];
end

nodePot(2, 1:action_n(2)) = 0;
nodePot(3, :) = 0;

for i = 1:nNodes
  nodePot(i, :) = exp(nodePot(i,:));
end
    
for i = 1:action_n(1)
  for j = 1:action_n(2)
    if floor((j - 1) / orient_n) + 1 ~= i
      edgePot(i,j,1) = -inf;
      continue;
    end
    idx = action_n(1) * featdims + (i - 1) * action_n(2) * featdims ...
          + (j - 1) * featdims + 1;
    edgePot(i,j,1) = w(idx:idx+featdims-1) * [feat_a2(1,j);feat_a2(2,j);1];
  end
end

idx_start = action_n(1) * featdims + action_n(1) * action_n(2) * featdims;
for i = 1:action_n(2)
    
  if i > 20
    edgePot(i,:,2) = 0;
    continue;
  end
    
  if i == 1
    y_start = 1;
  else
    y_start = sum(action_fine_n(1:i-1)) + 1;
  end
  y_end = sum(action_fine_n(1:i));
  
  for j = 1:action_n(3)
    if j < y_start || j > y_end
      edgePot(i,j,2) = -inf;
      continue;
    end
    idx = idx_start + (i -1) * action_n(3) * featdims + ...
           (j - 1) * featdims + 1;
    edgePot(i,j,2) = w(idx:idx+featdims-1) * [feat_a3(1,j);feat_a3(2,j);1];
  end
  
end

for i = 1:nEdges
  edgePot(:,:,i) = exp(edgePot(:,:,i));
end
    
label_pred = UGM_Decode_Tree(nodePot, edgePot, edgeStruct);
label_pred(2) = mod(label_pred(2)-1, action_n(1)) + 1;

