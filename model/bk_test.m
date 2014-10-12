function bk_test(conf)

data_test = conf.data_test;
w = conf.w;
delta_t = conf.delta_t;
clip_len = conf.clip_len;

action_n = [5 25 60];
orient_n = 5;
%action_fine_n = [2 2 3 2 1 2 3 4 2 1 3 3 1 2 2 3 3 1 2 2];
%R = 6;
%load('../subcategory/mat/model/ap_cluster.mat', 'pos_cluster_idx');
load('../subcategory/mat/pos_neg/data_l1.mat', 'state_idx');
for i = 1:action_n(1)-1
  for j = 1:orient_n
      
    k = (i-1) * orient_n + j;
    %action_fine_n(k) = length(pos_cluster_idx{i}{j});
    action_fine_n(k) = length(state_idx{i}{j});
  end
end
% for i = 1:action_n(1)-1
%   for j = 1:orient_n
%       
%     k = (i-1) * orient_n + j;
%     action_fine_n(k) = length(pos_cluster_idx{i}{j});
%     
%   end
% end

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

%   feat_a1{n} = [data_test(i).score_action; data_test(i).score_action_flow];
%   %feat_a1{i} = [data_train(i).score_action_flow; zeros(1, action_n(1))];
%   feat_a2{n} = [data_test(i).score_action_orient; ...
%                 data_test(i).score_action_orient_flow];
%   feat_a3{n} = [data_test(i).score_action_fine; ...
%                 data_test(i).score_action_fine_flow];
      
%   feat_a1{n} = data_test(i).score_action;
%   feat_a2{n} = data_test(i).score_action_orient;
%   feat_a3{n} = data_test(i).score_action_fine;
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

%   len = size(feat_a1_tmp, 1);
%   feat_a1_v(1,:) = feat_a1_tmp(end,:);
%   feat_a2_v(1,:) = feat_a2_tmp(end,:);
%   feat_a3_v(1,:) = feat_a3_tmp(end,:);
  
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
 
%save(['mat/result_' int2str(delta_t(end)) '.mat'], 'confmat', 'acc', 'acc_all');
  
  
  
%   %label_gt = label_f_gt;
% 
%   np = size(label{n}, 1);
%   
%   spatial = zeros(np);
%   
%   PV = [];
%   m = 0;
%   G = sqrt(dist2(data_test(n).bbox_face, data_test(n).bbox_face));
%   for j = 1:length(G)
%     for k = 1:length(G)
%       if j < k
%         m = m + 1;
%         PV(m, :) = [j k G(j, k)];
%       end
%     end
%   end
%   
%   if isempty(PV)
%     conn{n} = 0;
%   else
%     [val conn{n}] = kruskal(PV);
%   end
%   
%   for j = 1:np
%       
%     bbox_j = data_test(n).bbox_face(j,:);
%     step_w = bbox_j(3) * 2/3;
%     step_h = bbox_j(4) * 2;
%     cen_x = bbox_j(1) + bbox_j(3)/2;
%     cen_y = bbox_j(2) + bbox_j(4)/2;
%     bin_x = -2.5*step_w : step_w : 2.5*step_w;
%     bin_y = [-step_h/2 step_h/2]; 
%     
%     for k = 1:np
%         
%       if j == k
%         continue;
%       end
%       
%       bbox_k = data_test(n).bbox_face(k,:);
%       rel_xy = bbox_k(1:2) + bbox_k(3:4)/2 - [cen_x cen_y];
%       
%       if (rel_xy(1) < bin_x(1)) || (rel_xy(1) > bin_x(end)) || ...
%          (rel_xy(2) < bin_y(1)) || (rel_xy(2) > bin_y(2))
%         spatial(j,k) = 6;
%       else
%         spatial(j,k) = floor((rel_xy(1) + 2.5*step_w) / step_w) + 1;
%       end
%       
%     end
%     
%   end
%   
%   label_p_f = zeros(np, length(action_n));
%   for iter_i = 1:iter_n
%      
%     label_p = infer_label(w, label_p_f, action_n, orient_n, ...
%           feat_a1{n}, feat_a2{n}, feat_a3{n}, feat_o{n}, spatial, R, conn{n});
% 
% %     label_p_f = infer_label_f(w, label_p, label_f_gt_tmp, action_n, orient_n, ...
% %                            feat_a1{n}, feat_a2{n}, feat_a3{n}, feat_o{n});  
% 
%     %disp([int2str(iter_i) ':' int2str(label_p(3)) ':' int2str(label_p_f(3))]);
% 
%   end
%   
%   label_pred = [label_pred; label_p];
%   label_f_pred = [label_f_pred; label_p_f(idx_keep, :)];

%   label_pred(n,1) = label_p_1s(1);
%   label_pred(n,2) = (label_p_1s(1)-1) * label_n(2) + label_p_1s(2);  

  %[val idx] = max(feat_a_l0{n});
  %label_pred(n,1) = idx;
  %label_pred(n,2) = idx;

  %score_pred = zeros(label_n(1) * label_n(2), 1);

  %for i = 1:label_n(1)
  %  for j = 1:label_n(2)
      
  %    k = (i - 1) * label_n(2) + j;
      
  %    label_a = i;
  %    label_o = j;
  %    label_a_o = k;

  %    idx_a = (label_a - 1) * featdims_a + 1 : label_a * featdims_a;
  %    score_pred(k) = w(idx_a) * feat_a{n};

  %    idx_o_start = label_n(1) * featdims_a;
  %    idx_o = idx_o_start + (label_o - 1) * featdims_o + 1: ...
  %            idx_o_start + label_o * featdims_o;
  %    score_pred(k) = score_pred(k) + w(idx_o) * feat_o{n};
      %score_pred(k) = w(idx1) * feat;

  %    idx_a_o_start = idx_o_start + label_n(2) * featdims_o;
  %    idx_a_o = idx_a_o_start + (label_a_o - 1) * featdims_a_o + 1 : ...
  %              idx_a_o_start + label_a_o * featdims_a_o;
  %    score_pred(k) = score_pred(k) + w(idx_a_o)*feat_a_o{n};
    
      %idx2_start = label_n(1) * featdims_a;
      %idx2 = idx2_start + (label_l2 - 1) * featdims_a_o + 1 : ...
      %       idx2_start + label_l2 * featdims_a_o;       
      %score_pred(k) = score_pred(k) + w(idx2) * [feat_a_o{n}(label_l2); 1];
      %score_pred(k) = score_pred(k) + w(idx2) * feat_a_o{n};
    
  %  end
  %end
  
  %[val k_pred] = max(score_pred);
  %label_pred(n,1) = floor((k_pred - 1) / 5) + 1;
  %label_pred(n,2) = k_pred;

%end

% label_gt = label_f_gt;
% label_pred = label_f_pred;

% for n = 1:size(label_gt, 1)
% %     
% %   if label_gt(n,1) == 0
% %     continue;
% %   end
% 
%     
%   confmat_l1(label_gt(n,1), label_pred(n,1)) = ...
%                            confmat_l1(label_gt(n,1), label_pred(n,1)) + 1;
% 
%   confmat_l2(label_gt(n,2), label_pred(n,2)) = ...
%                         confmat_l2(label_gt(n,2), label_pred(n,2)) + 1;  
%                     
%   confmat_l3(label_gt(n,3), label_pred(n,3)) = ...
%                         confmat_l3(label_gt(n,3), label_pred(n,3)) + 1;  
% end
% 
% confmat_l1 = confmat_l1 + eps;
% confmat_l1 = confmat_l1 ./ repmat(sum(confmat_l1, 2), [1 action_n(1)]);
% acc_l1 = mean(diag(confmat_l1))
% acc_all_l1 = sum(label_gt(:,1) == label_pred(:,1)) / n
% 
% confmat_l2 = confmat_l2 + eps;
% confmat_l2 = confmat_l2 ./ repmat(sum(confmat_l2, 2), [1 size(confmat_l2, 1)]);
% acc_l2 = mean(diag(confmat_l2))
% acc_all_l2 = sum(label_gt(:,2) == label_pred(:,2)) / n
% 
% confmat_l3 = confmat_l3 + eps;
% confmat_l3 = confmat_l3 ./ repmat(sum(confmat_l3, 2), [1 size(confmat_l3, 1)]);
% acc_l3 = mean(diag(confmat_l3))
% acc_all_l3 = sum(label_gt(:,3) == label_pred(:,3)) / n
% 
% confmat_l2


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

% for i = 1:action_n(2)
%   idx = action_n(1) * 2 + (i - 1) * action_n(2) * 2 + (j - 1) * 2 + 1;
%   nodePot(2,i) = w(idx:idx+1) * [feat_a2(j); 1];
% end

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


% function label_pred = infer_label(w, label_f, action_n, orient_n, ...
%                       feat_a1, feat_a2, feat_a3, feat_o, spatial, R, conn)
% 
% np = size(feat_a1,1); % number of people per image                        
% label_n = length(action_n); % n labels for one person
% 
% nStates = [];
% for i = 1:np
%   nStates = [nStates action_n];
% end
% nNodes = length(nStates);
% maxState = max(nStates);
% 
% adj = zeros(nNodes);
% for i = 1:np
%   for j = 1:label_n-1
%     idx = (i-1)*label_n + j;
%     adj(idx,idx+1) = 1;
%   end
% end
% 
% for i = 1:np-1
%   for j = i+1:np
%     if conn(i, j) == 0
%       continue;
%     end
%     idx_i = i * label_n;
%     idx_j = j * label_n;
%     adj(idx_i, idx_j) = 1;
%   end
% end
% adj = adj + adj';
%     
% %%%%%%%%%%%%%%%%%%%%%%% compute unary potentials %%%%%%%%%%%%%%%%%%%%%%%%%%
% featdims_a = [size(feat_a1,2) size(feat_a2,2) size(feat_a3,2)];
% featdims_o = size(feat_o,2);
% 
% edgeStruct = UGM_makeEdgeStruct(adj, nStates);
% nEdges = edgeStruct.nEdges;
%     
% nodePot = -inf*ones(nNodes, maxState);
% edgePot = -inf*ones(maxState, maxState, nEdges);
% for p_i = 1:np
% 
% idx_f_start = featdims_a * action_n' + featdims_o * orient_n + ...
%               action_n(1) * action_n(2) + action_n(2) * action_n(3);
% 
% % level 1
% for i = 1:action_n(1)
%   idx = (i - 1) * featdims_a(1) + 1 : i * featdims_a(1);
%   nodePot((p_i-1)*label_n+1,i) = w(idx) * feat_a1(p_i,:)';                             
% end
% 
% if label_f(p_i,1) ~= 0
%   for i = 1:action_n(1)    
%     idx_f = idx_f_start + (i - 1) * action_n(1) + label_f(p_i,1);
%     nodePot((p_i-1)*label_n+1,i) = nodePot((p_i-1)*label_n+1,i) + w(idx_f);
%   end
% end
%   
% % level 2
% idx_start = action_n(1) * featdims_a(1);
% idx_f_start = idx_f_start + action_n(1)^2;
% for i = 1:action_n(2)
%   idx = idx_start + (i - 1) * featdims_a(2) + 1 : ...
%         idx_start + i * featdims_a(2);
%   nodePot((p_i-1)*label_n+2,i) = w(idx) * feat_a2(p_i,:)';
% end
% 
% if label_f(p_i,2) ~= 0
%   for i = 1:action_n(2)
%     idx_f = idx_f_start + (i-1)*action_n(2) + label_f(p_i,2);
%     nodePot((p_i-1)*label_n+2,i) = nodePot((p_i-1)*label_n+2,i) + w(idx_f);  
%   end
% end
% 
% % level 3
% idx_start = idx_start + action_n(2) * featdims_a(2);
% idx_start1 = idx_start + orient_n * featdims_o;
% idx_f_start = idx_f_start + action_n(2)^2;
% for i = 1:action_n(3)
%   j = mod(i - 1, action_n(2)) + 1; % orientation
%   idx = idx_start + (j - 1) * featdims_o + 1: ...
%         idx_start + j * featdims_o;
%   nodePot((p_i-1)*label_n+3,i) = w(idx) * feat_o(p_i,:)';
% 
% %   idx = idx_start1 + (i - 1) * featdims_a(3) + 1 : ...
% %         idx_start1 + i * featdims_a(3);
% %   nodePot((p_i-1)*label_n+3,i) = nodePot((p_i-1)*label_n+3,i) + ...
% %    w(idx) * feat_a3(p_i,:)' + loss_r(3) * (label(p_i,3)~=i) * action_r{3}(label(p_i,3));
% end
% 
% if label_f(p_i,3) ~= 0
%   for i = 1:action_n(3)
%     idx_f = idx_f_start + (i-1)*action_n(3) + label_f(p_i,3);
%     nodePot((p_i-1)*label_n+3,i) = nodePot((p_i-1)*label_n+3,i) + w(idx_f);
%   end
% end
% 
% node_i = (p_i-1)*label_n+1;
% node_j = (p_i-1)*label_n+2;
% edge_idx = find(ismember(edgeStruct.edgeEnds, [node_i node_j], 'rows'));
%     
% idx_start = idx_start1 + featdims_a(3) * action_n(3);
% for i = 1:action_n(1)
%   for j = 1:action_n(2)
%     idx = idx_start + (i-1) * action_n(2) + j;
%     edgePot(i,j,edge_idx) = w(idx);
%   end
% end
% 
% node_i = (p_i-1)*label_n+2;
% node_j = (p_i-1)*label_n+3;
% edge_idx = find(ismember(edgeStruct.edgeEnds, [node_i node_j], 'rows'));  
%     
% idx_start = idx_start + action_n(1) * action_n(2);
% for i = 1:action_n(2)
%   for j = 1:action_n(3)
%     
%     action_j = floor((j - 1) / action_n(2)) + 1;
%     if i ~= action_j
%       edgePot(i,j,edge_idx) = -inf;
%     else
%       idx = idx_start + (i-1) * action_n(3) + j;
%       edgePot(i,j,edge_idx) = w(idx);
%     end
%   end
% end
% 
% end % for p_i
% 
% for i = 1:nNodes
%   nodePot(i, :) = exp(nodePot(i,:));
% end
% 
% idx_start = featdims_a * action_n' + featdims_o * orient_n + ...
%             action_n(1) * action_n(2) + action_n(2) * action_n(3) + ...
%             action_n(1)^2 + action_n(2)^2 + action_n(3)^2;
%        
% for i = 1:np-1
%   for j = i+1:np
%       
%     if conn(i,j) == 0
%       continue;
%     end
% 
%     node_i = i * label_n;
%     node_j = j * label_n;
%     edge_idx = find(ismember(edgeStruct.edgeEnds, [node_i node_j], 'rows')); 
%           
%     for k = 1:action_n(3)
%       %action_k = floor((k - 1) / action_n(2)) + 1;
%       for m = 1:action_n(3)
%         %action_m = floor((m - 1) / action_n(2)) + 1;
%         idx = idx_start + (k-1)*action_n(3)*R + (m-1)*R + spatial(i,j);
%         idx1 = idx_start + (k-1)*action_n(3)*R + (m-1)*R + spatial(j,i);
%         edgePot(k,m,edge_idx) = w(idx) + w(idx1);
%       end
%     end
%       
%   end
% end
% 
% for i = 1:nEdges
%   edgePot(:,:,i) = exp(edgePot(:,:,i));
% end
%     
% label_pred_all = UGM_Decode_Tree(nodePot, edgePot, edgeStruct);
% 
% label_pred = zeros(np, label_n);
% for i = 1:np
%   idx = (i-1)*label_n+1:i*label_n;
%   label_pred(i,:) = label_pred_all(idx);
% end                        
                        
% np = size(feat_a1,1); % number of people per image                        
% label_n = length(action_n); % n labels for one person
% 
% nStates = [];
% for i = 1:np
%   nStates = [nStates action_n];
% end
% nNodes = length(nStates);
% maxState = max(nStates);
% 
% adj_ini = zeros(nNodes);
% for i = 1:np
%   for j = 1:label_n-1
%     idx = (i-1)*label_n + j;
%     adj_ini(idx,idx+1) = 1;
%   end
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%% compute unary potentials %%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% featdims_a = [size(feat_a1,2) size(feat_a2,2) size(feat_a3,2)];
% featdims_o = size(feat_o,2);
% 
% nEdges = np * 2;
% nodePot = -inf*ones(nNodes, maxState);
% edgePot = -inf*ones(maxState, maxState, nEdges);
% for p_i = 1:np
% 
% idx_f_start = featdims_a * action_n' + featdims_o * orient_n + ...
%               action_n(1) * action_n(2) + action_n(2) * action_n(3);
% 
% % level 1
% for i = 1:action_n(1)
%   idx = (i - 1) * featdims_a(1) + 1 : i * featdims_a(1);
%   nodePot((p_i-1)*label_n+1,i) = w(idx) * feat_a1(p_i,:)';                             
% end
% 
% if label_f(p_i,1) ~= 0
%   for i = 1:action_n(1)    
%     idx_f = idx_f_start + (i - 1) * action_n(1) + label_f(p_i,1);
%     nodePot((p_i-1)*label_n+1,i) = nodePot((p_i-1)*label_n+1,i) + w(idx_f);
%   end
% end
%   
% % level 2
% idx_start = action_n(1) * featdims_a(1);
% idx_f_start = idx_f_start + action_n(1)^2;
% for i = 1:action_n(2)
%   idx = idx_start + (i - 1) * featdims_a(2) + 1 : ...
%         idx_start + i * featdims_a(2);
%   nodePot((p_i-1)*label_n+2,i) = w(idx) * feat_a2(p_i,:)';
% end
% 
% if label_f(p_i,2) ~= 0
%   for i = 1:action_n(2)
%     idx_f = idx_f_start + (i-1)*action_n(2) + label_f(p_i,2);
%     nodePot((p_i-1)*label_n+2,i) = nodePot((p_i-1)*label_n+2,i) + w(idx_f);  
%   end
% end
% 
% % level 3
% idx_start = idx_start + action_n(2) * featdims_a(2);
% idx_start1 = idx_start + orient_n * featdims_o;
% idx_f_start = idx_f_start + action_n(2)^2;
% for i = 1:action_n(3)
%   j = mod(i - 1, action_n(2)) + 1; % orientation
%   idx = idx_start + (j - 1) * featdims_o + 1: ...
%         idx_start + j * featdims_o;
%   nodePot((p_i-1)*label_n+3,i) = w(idx) * feat_o(p_i,:)';
% 
% %   idx = idx_start1 + (i - 1) * featdims_a(3) + 1 : ...
% %         idx_start1 + i * featdims_a(3);
% %   nodePot((p_i-1)*label_n+3,i) = nodePot((p_i-1)*label_n+3,i) + ...
% %                  w(idx) * feat_a3(p_i,:)';
% end
% 
% if label_f(p_i,3) ~= 0
%   for i = 1:action_n(3)
%     idx_f = idx_f_start + (i-1)*action_n(3) + label_f(p_i,3);
%     nodePot((p_i-1)*label_n+3,i) = nodePot((p_i-1)*label_n+3,i) + w(idx_f);
%   end
% end
% 
% idx_start = idx_start1 + featdims_a(3) * action_n(3);
% for i = 1:action_n(1)
%   for j = 1:action_n(2)
%     idx = idx_start + (i-1) * action_n(2) + j;
%     edgePot(i,j,(p_i-1)*2+1) = w(idx);
%   end
% end
% 
% idx_start = idx_start + action_n(1) * action_n(2);
% for i = 1:action_n(2)
%   for j = 1:action_n(3)
%     action_j = floor((j - 1) / action_n(2)) + 1;
%     if i ~= action_j
%       edgePot(i,j,(p_i-1)*2+2) = -inf;
%     else
%       idx = idx_start + (i-1) * action_n(3) + j;
%       edgePot(i,j,(p_i-1)*2+2) = w(idx);
%     end
%   end
% end
% 
% end % for p_i
% 
% for i = 1:nNodes
%   nodePot(i,:) = exp(nodePot(i,:));
% end
% 
% edgePot_ini = edgePot;
% 
% %%%%%%%%%%%%%%%%%%%%%%%% compute pairwise potentials %%%%%%%%%%%%%%%%%%%%%%
% 
% % enumerate over all possible interactions
% inter_set = [];
% if np > 1 
%   for i = 1:np-1
%     for j = i+1:np
%       inter_set = [inter_set; [i j]];
%     end
%   end
% end
% inter_set = [inter_set; [0 0]];
% 
% pot = zeros(1, size(inter_set, 1));
% label_pred_all = cell(1, size(inter_set, 1));
% for inter_i = 1:size(inter_set, 1)
% 
%   adj = adj_ini;
%   %edgePot = edgePot_ini;
%   %edgePot = [];
%   if inter_set(inter_i,1) ~= 0
%     p_i = inter_set(inter_i,1);
%     p_j = inter_set(inter_i,2);
%     idx_i = p_i * label_n;
%     idx_j = p_j * label_n;
%     adj(idx_i, idx_j) = 1;
%     adj = adj + adj';
% 
%     edgeStruct = UGM_makeEdgeStruct(adj, nStates);
%     nEdges = edgeStruct.nEdges;
%     edge_idx = find(ismember(edgeStruct.edgeEnds, [idx_i idx_j], 'rows'));
%     edgePot = -inf*ones(maxState, maxState, nEdges);
% 
%     for i = 1:nEdges
%       if i < edge_idx
%         edgePot(:,:,i) = edgePot_ini(:,:,i);
%       elseif i > edge_idx
%         edgePot(:,:,i) = edgePot_ini(:,:,i-1);
%       else
%         idx_start = featdims_a * action_n' + featdims_o * orient_n + ...
%                     action_n(1) * action_n(2) + action_n(2) * action_n(3) + ...
%                     action_n(1)^2 + action_n(2)^2 + action_n(3)^2;
%              
%         for j = 1:action_n(3)
%           action_j = floor((j - 1) / action_n(2)) + 1;
%           for k = 1:action_n(3)
%             action_k = floor((k - 1) / action_n(2)) + 1;
%             if action_j ~= action_k || action_j == 5
%               edgePot(j,k,i) = -inf;
%             else
%               idx = idx_start + (j-1)*action_n(3)*R + (k-1)*R + spatial(p_i,p_j);
%               idx1 = idx_start + (j-1)*action_n(3)*R + (k-1)*R + spatial(p_j,p_i);
%               edgePot(j,k,i) = w(idx) + w(idx1);
%             end
%           end
%         end
%       end
%     end
% 
%   else
%     adj = adj + adj';
%     edgePot = edgePot_ini;
%     edgeStruct = UGM_makeEdgeStruct(adj, nStates);
%     nEdges = edgeStruct.nEdges;
%   end
%     
%   %edgeStruct = UGM_makeEdgeStruct(adj, nStates);
%   %nEdges = edgeStruct.nEdges;
%   
%   for i = 1:nEdges
%     edgePot(:,:,i) = exp(edgePot(:,:,i));
%   end
%     
%   [pot_tmp label_pred_all{inter_i}] = UGM_Decode_Tree(nodePot, ...
%                                                       edgePot,edgeStruct);
%   %pot(inter_i) = prod(pot_tmp);
%   pot(inter_i) = 1;
%   label = label_pred_all{inter_i};
% 
%   for i = 1:nNodes
%     pot(inter_i) = pot(inter_i) * nodePot(i,label(i));
%   end
% 
%   for i = 1:nEdges
%     edge_idx = edgeStruct.edgeEnds(i,:);
%     pot(inter_i) = pot(inter_i) * edgePot(label(edge_idx(1)), label(edge_idx(2)), i);
%   end
% 
% end
% 
% [val maxind] = max(pot);
% 
% label_pred = zeros(np, label_n);
% for i = 1:np
%   idx = (i-1)*label_n+1:i*label_n;
%   label_pred(i,:) = label_pred_all{maxind}(idx);
% 
%   %label_pred(i,:)
%   %pause;
% 
% %   
% %   if label_pred(i,2) > 5
% %     debug = 1;
% %   end
% %   
% 
% end




                        
%np = size(feat_a1,1);                        
% label_n = length(action_n);
% 
% nStates = [];
% for i = 1:np
%   nStates = [nStates action_n];
% end
% nNodes = length(nStates);
% maxState = max(nStates);
% 
% adj = zeros(nNodes);
% 
% for i = 1:np
%   for j = 1:label_n-1
%     idx = (i-1)*label_n + j;
%     adj(idx,idx+1) = 1;
%   end
% end
% adj = adj + adj';
% 
% edgeStruct = UGM_makeEdgeStruct(adj, nStates);
% nEdges = edgeStruct.nEdges;
% 
% featdims_a = [size(feat_a1,2) size(feat_a2,2) size(feat_a3,2)];
% featdims_o = size(feat_o,2);
% %featdims_a = [length(feat_a1) length(feat_a2) length(feat_a3)];
% %featdims_o = length(feat_o);
% 
% nodePot = zeros(nNodes, maxState);
% edgePot = zeros(maxState, maxState, nEdges);
% 
% for p_i = 1:np
% 
% idx_f_start = featdims_a * action_n' + featdims_o * orient_n + ...
%               action_n(1) * action_n(2) + action_n(2) * action_n(3);
% 
% % level 1
% for i = 1:action_n(1)
%   idx = (i - 1) * featdims_a(1) + 1 : i * featdims_a(1);
%   nodePot((p_i-1)*label_n+1,i) = w(idx) * feat_a1(p_i,:)';
% end
% 
% if label_f(p_i,1) ~= 0
%   for i = 1:action_n(1)    
%     idx_f = idx_f_start + (i - 1) * action_n(1) + label_f(p_i,1);
%     nodePot((p_i-1)*label_n+1,i) = nodePot((p_i-1)*label_n+1,i) + w(idx_f);
%   end
% end
%   
% % level 2
% idx_start = action_n(1) * featdims_a(1);
% idx_f_start = idx_f_start + action_n(1)^2;
% for i = 1:action_n(2)
%   idx = idx_start + (i - 1) * featdims_a(2) + 1 : ...
%         idx_start + i * featdims_a(2);
%   nodePot((p_i-1)*label_n+2,i) = w(idx) * feat_a2(p_i,:)';
%   %nodePot{2}(i) = feat_a2(i);
% end
% 
% if label_f(p_i,2) ~= 0
%   for i = 1:action_n(2)
%     idx_f = idx_f_start + (i-1)*action_n(2) + label_f(p_i,2);
%     nodePot((p_i-1)*label_n+2,i) = nodePot((p_i-1)*label_n+2,i) + w(idx_f);  
%   end
% end
% 
% % level 3
% idx_start = idx_start + action_n(2) * featdims_a(2);
% idx_start1 = idx_start + orient_n * featdims_o;
% idx_f_start = idx_f_start + action_n(2)^2;
% for i = 1:action_n(3)
%   j = mod(i - 1, action_n(2)) + 1; % orientation
%   idx = idx_start + (j - 1) * featdims_o + 1: ...
%         idx_start + j * featdims_o;
%   nodePot((p_i-1)*label_n+3,i) = w(idx) * feat_o(p_i,:)';
% 
%   idx = idx_start1 + (i - 1) * featdims_a(3) + 1 : ...
%         idx_start1 + i * featdims_a(3);
%   nodePot((p_i-1)*label_n+3,i) = nodePot((p_i-1)*label_n+3,i) + ...
%                                  w(idx) * feat_a3(p_i,:)';  
% end
% 
% if label_f(p_i,3) ~= 0
%   for i = 1:action_n(3)
%     idx_f = idx_f_start + (i-1)*action_n(3) + label_f(p_i,3);
%     nodePot((p_i-1)*label_n+3,i) = nodePot((p_i-1)*label_n+3,i) + w(idx_f);
%   end
% end
% 
% idx_start = idx_start1 + featdims_a(3) * action_n(3);
% 
% for i = 1:action_n(1)
%   for j = 1:action_n(2)
%     idx = idx_start + (i-1) * action_n(2) + j;
%     %edgePot(i,j,1) = exp(w(idx));
%     edgePot(i,j,(p_i-1)*2+1) = w(idx);
%   end
% end
% 
% idx_start = idx_start + action_n(1) * action_n(2);
% %gamma = zeros(action_n(2), action_n(3));
% for i = 1:action_n(2)
%   for j = 1:action_n(3)
%     action_j = floor((j - 1) / action_n(2)) + 1;
%     if i ~= action_j
%       edgePot(i,j,(p_i-1)*2+2) = -inf;
%     else
%       idx = idx_start + (i-1) * action_n(3) + j;
%       edgePot(i,j,(p_i-1)*2+2) = w(idx);
%     end
%   end
% end
% 
% end
% 
% for i = 1:nNodes
%   nodePot(i, :) = exp(nodePot(i,:));
% end
% 
% for i = 1:nEdges
%   edgePot(:,:,i) = exp(edgePot(:,:,i));
% end
% 
% %fprintf('Decoding: Finding configuration of variables with highest potentials...\n');
% label_pred_all = UGM_Decode_Tree(nodePot,edgePot,edgeStruct);
% 
% label_pred = zeros(np, label_n);
% for i = 1:np
%   idx = (i-1)*label_n+1:i*label_n;
%   label_pred(i,:) = label_pred_all(idx);
% end


% nStates = action_n;
% nNodes = length(nStates);
% maxState = max(nStates);
% adj = zeros(nNodes);
% for i = 1:nNodes-1
%   adj(i,i+1) = 1;
% end
% adj = adj + adj';
% 
% edgeStruct = UGM_makeEdgeStruct(adj, nStates);
% nEdges = edgeStruct.nEdges;
% 
% featdims_a = [length(feat_a1) length(feat_a2) length(feat_a3)];
% featdims_o = length(feat_o);
% 
% nodePot = zeros(nNodes, maxState);
% 
% idx_f_start = featdims_a * action_n' + featdims_o * orient_n + ...
%               action_n(1) * action_n(2) + action_n(2) * action_n(3);
% % level 1
% for i = 1:action_n(1)
%   idx = (i - 1) * featdims_a(1) + 1 : i * featdims_a(1);
%   idx_f = idx_f_start + (i - 1) * action_n(1) + label_f(1);
%   nodePot(1,i) = w(idx) * feat_a1 + w(idx_f);
% end
%   
% % level 2
% idx_start = action_n(1) * featdims_a(1);
% idx_f_start = idx_f_start + action_n(1)^2;
% for i = 1:action_n(2)
%   idx = idx_start + (i - 1) * featdims_a(2) + 1 : ...
%         idx_start + i * featdims_a(2);
%   idx_f = idx_f_start + (i-1)*action_n(2) + label_f(2);
%   nodePot(2,i) = w(idx) * feat_a2 + w(idx_f);
%   %nodePot{2}(i) = feat_a2(i);
% end
% 
% % level 3
% idx_start = idx_start + action_n(2) * featdims_a(2);
% idx_start1 = idx_start + orient_n * featdims_o;
% idx_f_start = idx_f_start + action_n(2)^2;
% for i = 1:action_n(3)
%   j = mod(i - 1, action_n(2)) + 1; % orientation
%   idx = idx_start + (j - 1) * featdims_o + 1: ...
%         idx_start + j * featdims_o;
%   nodePot(3,i) = w(idx) * feat_o;
% 
%   idx = idx_start1 + (i - 1) * featdims_a(3) + 1 : ...
%       idx_start1 + i * featdims_a(3);
%   nodePot(3,i) = nodePot(3,i) + w(idx) * feat_a3;
%   
%   idx_f = idx_f_start + (i-1)*action_n(3) + label_f(3);
%   nodePot(3,i) = nodePot(3,i) + w(idx_f);
% end
% 
% for i = 1:nNodes
%   nodePot(i, :) = exp(nodePot(i,:));
% end
% 
% idx_start = idx_start1 + featdims_a(3) * action_n(3);
% edgePot = zeros(maxState, maxState, nEdges);
% 
% for i = 1:action_n(1)
%   for j = 1:action_n(2)
%     idx = idx_start + (i-1) * action_n(2) + j;
%     edgePot(i,j,1) = w(idx);
%   end
% end
% 
% idx_start = idx_start + action_n(1) * action_n(2);
% %gamma = zeros(action_n(2), action_n(3));
% for i = 1:action_n(2)
%   for j = 1:action_n(3)
%     action_j = floor((j - 1) / action_n(2)) + 1;
%     if i ~= action_j
%       edgePot(i,j,2) = -inf;
%     else
%       idx = idx_start + (i-1) * action_n(3) + j;
%       edgePot(i,j,2) = w(idx);
%     end
%   end
% end
% 
% for i = 1:nEdges
%   edgePot(:,:,i) = exp(edgePot(:,:,i));
% end
% 
% % score_pred = zeros(action_n(1), action_n(2), action_n(3));
% % for i = 1:action_n(1)
% %   for j = 1:action_n(2)
% %     for k = 1:action_n(3)
% %       score_pred(i,j,k) = nodePot(1,i) * nodePot(2,j) * nodePot(3,k) * ...
% %                           edgePot(i,j,1) * edgePot(j,k,2);
% %     end
% %   end
% % end
% % 
% % [val idx] = max(score_pred(:));
% % [label_pred(1) label_pred(2) label_pred(3)] = ind2sub(action_n, idx);
% 
% % fprintf('Decoding: Finding configuration of variables with highest potentials...\n');
% label_pred = UGM_Decode_Tree(nodePot,edgePot,edgeStruct);

% nodePot = cell(1, nNodes);
% edgePot = cell(nNodes, nNodes);
% conn = zeros(nNodes, nNodes);
% for i = 1:nNodes-1
%   conn(i,i+1) = 1;
%   conn(i+1,i) = 1;
% end
% 
% for i = 1:nNodes
%   nodePot{i} = zeros(action_n(i), 1);
% end
% 
% featdims_a = [length(feat_a1) length(feat_a2) length(feat_a3)];
% featdims_o = length(feat_o);
% 
% idx_f_start = featdims_a * action_n' + featdims_o * orient_n + ...
%               action_n(1) * action_n(2) + action_n(2) * action_n(3);
% % level 1
% for i = 1:action_n(1)
%   idx = (i - 1) * featdims_a(1) + 1 : i * featdims_a(1);
%   idx_f = idx_f_start + (i - 1) * action_n(1) + label_f(1);
%   nodePot{1}(i) = w(idx) * feat_a1 + w(idx_f);
% end
%   
% % level 2
% idx_start = action_n(1) * featdims_a(1);
% idx_f_start = idx_f_start + action_n(1)^2;
% for i = 1:action_n(2)
%   idx = idx_start + (i - 1) * featdims_a(2) + 1 : ...
%         idx_start + i * featdims_a(2);
%   idx_f = idx_f_start + (i-1)*action_n(2) + label_f(2);
%   nodePot{2}(i) = w(idx) * feat_a2 + w(idx_f);
%   %nodePot{2}(i) = feat_a2(i);
% end
% 
% % level 3
% idx_start = idx_start + action_n(2) * featdims_a(2);
% idx_start1 = idx_start + orient_n * featdims_o;
% idx_f_start = idx_f_start + action_n(2)^2;
% for i = 1:action_n(3)
%   j = mod(i - 1, action_n(2)) + 1; % orientation
%   idx = idx_start + (j - 1) * featdims_o + 1: ...
%         idx_start + j * featdims_o;
%   nodePot{3}(i) = w(idx) * feat_o;
% 
%   idx = idx_start1 + (i - 1) * featdims_a(3) + 1 : ...
%       idx_start1 + i * featdims_a(3);
%   nodePot{3}(i) = nodePot{3}(i) + w(idx) * feat_a3;
%   
%   idx_f = idx_f_start + (i-1)*action_n(3) + label_f(3);
%   nodePot{3}(i) = nodePot{3}(i) + w(idx_f);
% end
% 
% idx_start = idx_start1 + featdims_a(3) * action_n(3);
% gamma = zeros(action_n(1), action_n(2));
% for i = 1:action_n(1)
%   for j = 1:action_n(2)
%     idx = idx_start + (i-1) * action_n(2) + j;
%     gamma(i,j) = w(idx);
%   end
% end
% edgePot{1,2} = exp(gamma);
% edgePot{2,1} = exp(gamma);
% 
% idx_start = idx_start + action_n(1) * action_n(2);
% gamma = zeros(action_n(2), action_n(3));
% for i = 1:action_n(2)
%   for j = 1:action_n(3)
%     action_j = floor((j - 1) / action_n(2)) + 1;
%     if i ~= action_j
%       gamma(i,j) = -inf;
%     else
%       idx = idx_start + (i-1) * action_n(3) + j;
%       gamma(i,j) = w(idx);
%     end
%   end
% end
% edgePot{2,3} = exp(gamma);
% edgePot{3,2} = exp(gamma);
% 
% for i = 1:nNodes
%   nodePot{i} = exp(nodePot{i});
% end
% 
% [bel, converged, belpairs, msgs] = inference(conn,edgePot,nodePot,'loopy','sum_or_max',1,...
%                              'max_iter',1);
% 
% label_pred1 = zeros(1, nNodes);                         
% for i = 1:nNodes
%   [val, label_pred1(i)] = max(bel{i});
% end
% 
% label_pred
% label_pred1

% for i = 1:nNodes
%   [val, label_pred(i)] = max(nodePot{i});
% end



function label_pred = infer_label_f(w, label, label_gt, action_n, orient_n, ...
                            feat_a1, feat_a2, feat_a3, feat_o)

np = size(feat_a1,1);                        
label_n = length(action_n);

nStates = [];
for i = 1:np
  nStates = [nStates action_n];
end
nNodes = length(nStates);
maxState = max(nStates);

adj = zeros(nNodes);

for i = 1:np
  for j = 1:label_n-1
    idx = (i-1)*label_n + j;
    adj(idx,idx+1) = 1;
  end
end
adj = adj + adj';

edgeStruct = UGM_makeEdgeStruct(adj, nStates);
nEdges = edgeStruct.nEdges;

featdims_a = [size(feat_a1,2) size(feat_a2,2) size(feat_a3,2)];
featdims_o = size(feat_o,2);
%featdims_a = [length(feat_a1) length(feat_a2) length(feat_a3)];
%featdims_o = length(feat_o);

% nodePot = zeros(nNodes, maxState);
% edgePot = zeros(maxState, maxState, nEdges);

nodePot = -inf*ones(nNodes, maxState);
edgePot = -inf*ones(maxState, maxState, nEdges);

for p_i = 1:np
    
  if label_gt(p_i, 1) == 0
    continue;
  end    

idx_f_start = featdims_a * action_n' + featdims_o * orient_n + ...
              action_n(1) * action_n(2) + action_n(2) * action_n(3);

% level 1
for i = 1:action_n(1)
  idx_f = idx_f_start + (label(p_i,1) - 1) * action_n(1) + i;
  nodePot((p_i-1)*label_n+1,i) = w(idx_f);
end
  
% level 2
idx_f_start = idx_f_start + action_n(1)^2;
for i = 1:action_n(2)
  idx_f = idx_f_start + (label(p_i,2)-1)*action_n(2) + i;
  nodePot((p_i-1)*label_n+2,i) = w(idx_f);
end

% level 3
idx_f_start = idx_f_start + action_n(2)^2;
for i = 1:action_n(3)  
  idx_f = idx_f_start + (label(p_i,3)-1)*action_n(3) + i;
  nodePot((p_i-1)*label_n+3,i) = w(idx_f);
end

idx_f_start = featdims_a * action_n' + featdims_o * orient_n;

for i = 1:action_n(1)
  for j = 1:action_n(2)
    idx = idx_f_start + (i-1) * action_n(2) + j;
    edgePot(i,j,(p_i-1)*2+1) = w(idx);
  end
end

idx_f_start = idx_f_start + action_n(1) * action_n(2);
%gamma = zeros(action_n(2), action_n(3));
for i = 1:action_n(2)
  for j = 1:action_n(3)
    action_j = floor((j - 1) / action_n(2)) + 1;
    if i ~= action_j
      edgePot(i,j,(p_i-1)*2+2) = -inf;
    else
      idx = idx_f_start + (i-1) * action_n(3) + j;
      edgePot(i,j,(p_i-1)*2+2) = w(idx);
    end
  end
end

end

for i = 1:nNodes
  nodePot(i, :) = exp(nodePot(i,:));
end

for i = 1:nEdges
  edgePot(:,:,i) = exp(edgePot(:,:,i));
end

%fprintf('Decoding: Finding configuration of variables with highest potentials...\n');
[pot label_pred_all] = UGM_Decode_Tree(nodePot,edgePot,edgeStruct);

label_pred = zeros(np, label_n);
for i = 1:np
  if label_gt(i, 1) == 0
    continue;
  end  
  idx = (i-1)*label_n+1:i*label_n;
  label_pred(i,:) = label_pred_all(idx);
end


% nStates = action_n;
% nNodes = length(nStates);
% maxState = max(nStates);
% adj = zeros(nNodes);
% for i = 1:nNodes-1
%   adj(i,i+1) = 1;
% end
% adj = adj + adj';
% 
% edgeStruct = UGM_makeEdgeStruct(adj, nStates);
% nEdges = edgeStruct.nEdges;
% 
% featdims_a = [length(feat_a1) length(feat_a2) length(feat_a3)];
% featdims_o = length(feat_o);
% 
% nodePot = zeros(nNodes, maxState);
% 
% idx_f_start = featdims_a * action_n' + featdims_o * orient_n + ...
%               action_n(1) * action_n(2) + action_n(2) * action_n(3);
% % level 1
% for i = 1:action_n(1)
%   idx_f = idx_f_start + (label(1) - 1) * action_n(1) + i;
%   nodePot(1,i) = w(idx_f);
% end
%   
% % level 2
% idx_f_start = idx_f_start + action_n(1)^2;
% for i = 1:action_n(2)
%   idx_f = idx_f_start + (label(2)-1)*action_n(2) + i;
%   nodePot(2,i) = w(idx_f);
%   %nodePot{2}(i) = feat_a2(i);
% end
% 
% % level 3
% idx_f_start = idx_f_start + action_n(2)^2;
% for i = 1:action_n(3)  
%   idx_f = idx_f_start + (label(3)-1)*action_n(3) + i;
%   nodePot(3,i) = w(idx_f);
% end
% 
% for i = 1:nNodes
%   nodePot(i, :) = exp(nodePot(i,:));
% end
% 
% %idx_f_start = idx_f_start + action_n(3)^2;
% idx_f_start = featdims_a * action_n' + featdims_o * orient_n;
% edgePot = zeros(maxState, maxState, nEdges);
% 
% for i = 1:action_n(1)
%   for j = 1:action_n(2)
%     idx = idx_f_start + (i-1) * action_n(2) + j;
%     %edgePot(i,j,1) = exp(w(idx));
%     edgePot(i,j,1) = w(idx);
%   end
% end
% 
% idx_f_start = idx_f_start + action_n(1) * action_n(2);
% for i = 1:action_n(2)
%   for j = 1:action_n(3)
%     action_j = floor((j - 1) / action_n(2)) + 1;
%     if i ~= action_j
%       edgePot(i,j,2) = -inf;
%     else
%       idx = idx_f_start + (i-1) * action_n(3) + j;
%       edgePot(i,j,2) = w(idx);
%     end
%   end
% end
% 
% for i = 1:nEdges
%   edgePot(:,:,i) = exp(edgePot(:,:,i));
% end
% 
% %fprintf('Decoding: Finding configuration of variables with highest potentials...\n');
% label_pred = UGM_Decode_Tree(nodePot,edgePot,edgeStruct);
% label_pred = label_pred';             

% nNodes = 3;     
% 
% nodePot = cell(1, nNodes);
% edgePot = cell(nNodes, nNodes);
% conn = zeros(nNodes, nNodes);
% for i = 1:nNodes-1
%   conn(i,i+1) = 1;
%   conn(i+1,i) = 1;
% end
% 
% for i = 1:nNodes
%   nodePot{i} = zeros(action_n(i), 1);
% end
% 
% featdims_a = [length(feat_a1) length(feat_a2) length(feat_a3)];
% featdims_o = length(feat_o);
% 
% idx_f_start = featdims_a * action_n' + featdims_o * orient_n + ...
%               action_n(1) * action_n(2) + action_n(2) * action_n(3);
% % level 1
% for i = 1:action_n(1)
%   idx_f = idx_f_start + (label(1) - 1) * action_n(1) + i;
%   nodePot{1}(i) = w(idx_f);
% end
%   
% % level 2
% idx_f_start = idx_f_start + action_n(1)^2;
% for i = 1:action_n(2)
%   idx_f = idx_f_start + (label(2) - 1) * action_n(2) + i;
%   nodePot{2}(i) = w(idx_f);
% end
% 
% % level 3
% idx_f_start = idx_f_start + action_n(2)^2;
% for i = 1:action_n(3)
%   idx_f = idx_f_start + (label(3) - 1) * action_n(3) + i;
%   nodePot{3}(i) = w(idx_f);
% end
% 
% idx_f_start = idx_f_start + action_n(3)^2;
% gamma = zeros(action_n(1), action_n(2));
% for i = 1:action_n(1)
%   for j = 1:action_n(2)
%     idx = idx_f_start + (i-1) * action_n(2) + j;
%     gamma(i,j) = w(idx);
%   end
% end
% edgePot{1,2} = gamma;
% edgePot{2,1} = gamma;
% 
% idx_f_start = idx_f_start + action_n(1) * action_n(2);
% gamma = zeros(action_n(2), action_n(3));
% for i = 1:action_n(2)
%   for j = 1:action_n(3)
%     idx = idx_f_start + (i-1) * action_n(3) + j;
%     gamma(i,j) = w(idx);
%   end
% end
% edgePot{2,3} = gamma;
% edgePot{3,2} = gamma;
% 
% [bel, converged] = inference(conn,edgePot,nodePot,'loopy','sum_or_max',1,...
%                              'max_iter',1);
% 
% label_pred = zeros(1, nNodes);                         
% for i = 1:nNodes
%   [val, label_pred(i)] = max(bel{i});
% end

% function label_pred = infer_label(w, label_1s, label_n, feat_a, ...
%                                   feat_o, feat_a_o)
% 
% %label = label_1s;
% score_pred = zeros(label_n(1) * label_n(2), 1);
% featdims_a = length(feat_a);
% featdims_o = length(feat_o);
% featdims_a_o = length(feat_a_o);
% %featdims_a_o = 2;
% %label_gt_a = label(1);
% %label_gt_a_o = (label(1)-1)*label_n(2) + label(2);
% 
% for i = 1:label_n(1)
%   for j = 1:label_n(2)
%       
%     k = (i - 1) * label_n(2) + j;
%       
%     label_a = i;
%     label_o = j;
%     label_a_o = k;
% 
%     idx_a = (label_a - 1) * featdims_a + 1 : label_a * featdims_a;
%     score_pred(k) = w(idx_a) * feat_a;
% 
%     idx_o_start = label_n(1) * featdims_a;
%     idx_o = idx_o_start + (label_o - 1) * featdims_o + 1: ...
%             idx_o_start + label_o * featdims_o;
%     score_pred(k) = score_pred(k) + w(idx_o) * feat_o;
%  
%     idx_a_o_start = idx_o_start + label_n(2) * featdims_o;
%     idx_a_o = idx_a_o_start + (label_a_o - 1) * featdims_a_o + 1 : ...
%               idx_a_o_start + label_a_o * featdims_a_o;
%     score_pred(k) = score_pred(k) + w(idx_a_o)*feat_a_o;
%  
%     % action - action @ future
%     idx_f_a_start = idx_a_o_start + featdims_a_o*label_n(1)*label_n(2);
%     idx_f_a = idx_f_a_start + (label_a-1)*label_n(1) + label_1s(1);
%     score_pred(k) = score_pred(k) + w(idx_f_a);
% 
%     % action+orientation - action+orientation@future
%     idx_f_a_o_start = idx_f_a_start + label_n(1)^2;
%     label_a_o_1s = (label_1s(1)-1) * label_n(2) + label_1s(2);
%     idx_f_a_o = idx_f_a_o_start + (label_a_o-1)*label_n(1)^2 + label_a_o_1s;
%     score_pred(k) = score_pred(k) + w(idx_f_a_o);
% 
%   end
% end
% 
% [val k_pred] = max(score_pred);
% label_pred(1) = floor((k_pred - 1) / 5) + 1;
% label_pred(2) = mod(k_pred - 1, 5) + 1;
% %label_pred(2) = k_pred;
% 
% 
% function label_pred = infer_label_1s(w, label_prev, label_n, feat_a, ...
%                                   feat_o, feat_a_o)
% 
% score_pred = zeros(label_n(1) * label_n(2), 1);
% featdims_a = length(feat_a);
% featdims_o = length(feat_o);
% featdims_a_o = length(feat_a_o);
% %featdims_a_o = 2;
% %label_gt_a = label(1);
% %label_gt_a_o = (label(1)-1)*label_n(2) + label(2);
% %label_gt_a_o = (label(1)-1)*label_n(2) + label(2);
% 
% label_prev_a_o = (label_prev(1)-1)*label_n(2) + label_prev(2);
% 
% for i = 1:label_n(1)
%   for j = 1:label_n(2)
%       
%     k = (i - 1) * label_n(2) + j;
%       
%     label_a = i;
%     label_o = j;
%     label_a_o = k;
% 
%     %idx_a = (label_a - 1) * featdims_a + 1 : label_a * featdims_a;
%     %score_pred(k) = w(idx_a) * feat_a;
% 
%     %idx_o_start = label_n(1) * featdims_a;
%     %idx_o = idx_o_start + (label_o - 1) * featdims_o + 1: ...
%     %        idx_o_start + label_o * featdims_o;
%     %score_pred(k) = score_pred(k) + w(idx_o) * feat_o;
%  
%     %idx_a_o_start = idx_o_start + label_n(2) * featdims_o;
%     %idx_a_o = idx_a_o_start + (label_a_o - 1) * featdims_a_o + 1 : ...
%     %          idx_a_o_start + label_a_o * featdims_a_o;
%     %score_pred(k) = score_pred(k) + w(idx_a_o)*feat_a_o + ...
%     %                (label_gt_a_o ~= label_a_o);
%  
%     % action - action @ future
%     idx_f_a_start = label_n(1) * featdims_a + label_n(2) * featdims_o ...
%                     + featdims_a_o*label_n(1)*label_n(2);
%     idx_f_a = idx_f_a_start + (label_prev(1)-1)*label_n(1) + label_a;
%     score_pred(k) = score_pred(k) + w(idx_f_a);
% 
%     % action+orientation - action+orientation@future
%     idx_f_a_o_start = idx_f_a_start + label_n(1)^2;
%     idx_f_a_o = idx_f_a_o_start + (label_prev_a_o-1)*label_n(1)^2 + label_a_o;
%     score_pred(k) = score_pred(k) + w(idx_f_a_o);
% 
%   end
% end
% 
% [val k_pred] = max(score_pred);
% label_pred(1) = floor((k_pred - 1) / 5) + 1;
% label_pred(2) = mod(k_pred - 1, 5) + 1;
    
  
