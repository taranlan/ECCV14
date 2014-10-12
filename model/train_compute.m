function  [fvalue gradient nothing] = train_compute(x0, auxdata)
%compute sub-gradient

action_n = auxdata.action_n;
orient_n = auxdata.orient_n;
action_fine_n = auxdata.action_fine_n;
totaldims = auxdata.totaldims;
%action_r = auxdata.action_r;

gradient = zeros(totaldims, 1);
fvalue = 0;

%loss_r = [0 1 1]/6;

img_n = length(auxdata); 
clip_idx = auxdata(1).clip.idx;
video_n = length(clip_idx);

disp('computing sub-gradient');
tic

% for i = 1:img_n
%   %disp(['image: ' int2str(i) ' : ' int2str(img_n)]);
%   
%   label = auxdata(i).label;
%   feat_a1 = auxdata(i).feat_a1;
%   feat_a2 = auxdata(i).feat_a2;
%   feat_a3 = auxdata(i).feat_a3;
%   
%   phi_positive = construct_phi(label, action_n, orient_n, ...
%                                feat_a1, feat_a2, feat_a3, totaldims);
%   
%   fvalue_positive = x0 * phi_positive;
% 
%   label_pred = infer_label(x0, label, action_n, orient_n, action_fine_n, ...
%                            feat_a1, feat_a2, feat_a3);
%     
% 
%   phi_negative = construct_phi(label_pred, action_n, orient_n, ...
%                                feat_a1, feat_a2, feat_a3, totaldims);
%   
%   fvalue_negative = x0 * phi_negative + (label(1) ~= label_pred(1));
% 
%   curr_gradient = phi_negative - phi_positive;
%   curr_fvalue = fvalue_negative - fvalue_positive; 
%   
%   if curr_fvalue < 0
%    curr_fvalue = 0;
%    curr_gradient = zeros(totaldims, 1);
%   end
%   
%   gradient  = gradient + curr_gradient;
%   fvalue = fvalue + curr_fvalue; 
%   
% end

for i = 1:video_n  
  
  feat_a1_tmp = [];
  feat_a2_tmp = [];
  feat_a3_tmp = [];
  for j = clip_idx{i}(1) : clip_idx{i}(2)
      feat_a1_tmp = [feat_a1_tmp; auxdata(j).feat_a1(1,:)];
      feat_a2_tmp = [feat_a2_tmp; auxdata(j).feat_a2(1,:)];
      feat_a3_tmp = [feat_a3_tmp; auxdata(j).feat_a3(1,:)];
  end
  
  feat_a1(1,:) = max(feat_a1_tmp, [], 1);
  %feat_a3 = [];
  feat_a2(1,:) = max(feat_a2_tmp, [], 1);
  feat_a3(1,:) = max(feat_a3_tmp, [], 1);
  
  feat_a1_tmp = [];
  feat_a2_tmp = [];
  feat_a3_tmp = [];
  flow_n = 0;
  for j = clip_idx{i}(1) : clip_idx{i}(2)
    flow_n = flow_n + auxdata(j).flow_n;  
    feat_a1_tmp = [feat_a1_tmp; auxdata(j).feat_a1(2,:)];
    feat_a2_tmp = [feat_a2_tmp; auxdata(j).feat_a2(2,:)];
    feat_a3_tmp = [feat_a3_tmp; auxdata(j).feat_a3(2,:)];
  end
  
  feat_a1(2,:) = sum(feat_a1_tmp, 1) / (flow_n+eps);
  feat_a2(2,:) = sum(feat_a2_tmp, 1) / (flow_n+eps);
  feat_a3(2,:) = sum(feat_a3_tmp, 1) / (flow_n+eps);
  
  label = auxdata(clip_idx{i}(2)).label;
  
  %label = infer_label_l3(x0, label, action_n, action_fine_n, feat_a3);
  
  phi_positive = construct_phi(label, action_n, orient_n, ...
                               feat_a1, feat_a2, feat_a3, totaldims);
  fvalue_positive = x0 * phi_positive;

  label_pred = infer_label(x0, label, action_n, orient_n, action_fine_n, ...
                           feat_a1, feat_a2, feat_a3);
       
  phi_negative = construct_phi(label_pred, action_n, orient_n, ...
                               feat_a1, feat_a2, feat_a3, totaldims);

  fvalue_negative = x0 * phi_negative + (label(1) ~= label_pred(1));

  curr_gradient = phi_negative - phi_positive;
  curr_fvalue = fvalue_negative - fvalue_positive; 
  
  if curr_fvalue < 0
   curr_fvalue = 0;
   curr_gradient = zeros(totaldims, 1);
  end
  
  gradient  = gradient + curr_gradient;
  fvalue = fvalue + curr_fvalue;   
    
end


gradient = gradient';

nothing = [];
toc
disp('subgradient end');


function phi = construct_phi(label, action_n, orient_n, ...
                             feat_a1, feat_a2, feat_a3, totaldims)

phi = zeros(totaldims, 1);
featdims = 3;
%featdims_a = [length(feat_a1) length(feat_a2)];

y1 = label(1);
y2 = (y1 - 1) * orient_n + label(2);
y3 = label(3);

% level 1
% idx = (y1 - 1) * featdims_a(1) + 1 : y1 * featdims_a(1);
% phi(idx) = phi(idx) + feat_a1';
idx = (y1 - 1) * featdims + 1;
phi(idx:idx+featdims-1) = phi(idx:idx+featdims-1) + [feat_a1(1,y1); ...
                          feat_a1(2,y1); 1];

idx_start = action_n(1) * featdims;
idx = idx_start + (y1 - 1) * action_n(2) * featdims + (y2 - 1) * featdims + 1;
phi(idx:idx+featdims-1) = phi(idx:idx+featdims-1) + ...
                          [feat_a2(1,y2); feat_a2(2,y2); 1];

if y1 ~= 5
  idx_start = action_n(1) * featdims + action_n(1) * action_n(2) * featdims;
  idx = idx_start + (y2 -1) * action_n(3) * featdims + (y3 - 1) * featdims + 1;
  
  phi(idx:idx+featdims-1) = phi(idx:idx+featdims-1) + ...
                            [feat_a3(1,y3); feat_a3(2,y3); 1];
end

% function label = infer_label_l3(w, label, action_n, ...
%                                      action_fine_n, feat_a3)
%                                  
% pot = zeros(1, action_n(3));
% featdims = 3;  
% idx_start = action_n(1) * featdims + action_n(1) * action_n(2) * featdims;
% 
% if label(1) == 5
%     
%   label(3) = 0;
%   
% else
%     
%   if label(2) == 1
%     y_start = 1;
%   else
%     y_start = sum(action_fine_n(1:label(2)-1)) + 1;
%   end
%   y_end = sum(action_fine_n(1:label(2)));
%   
%   for j = 1:action_n(3)
%     if j < y_start || j > y_end
%       pot(j) = -inf;
%       continue;
%     end
%     idx = idx_start + (label(2) -1) * action_n(3) * featdims + ...
%           (j - 1) * featdims + 1;
%     pot(j) = w(idx:idx+featdims-1) * [feat_a3(1,j); feat_a3(2,j); 1];
%   
%     if sum(w(idx:idx+featdims-1) ~= 0)
%       debug = 1;
%     end
%   
%   end
%  
%   [val label(3)] = max(pot);
%   
%   
% end

function label_pred = infer_label(w, label, action_n, orient_n, action_fine_n, ...
                                  feat_a1, feat_a2, feat_a3, action_r)
    
% featdims_a = [length(feat_a1) length(feat_a2)];  
% score_pred = zeros(1, action_n(1));
% orient_pred = zeros(1, orient_n);
% for i = 1:action_n(1)
%    
% %   idx = (i - 1) * featdims_a + 1 : i * featdims_a;
% %   score_pred(i) = w(idx) * feat_a1' + (label(1) ~= i);
% 
%   idx = (i - 1) * 2 + 1;
%   score_pred(i) = w(idx:idx+1) * [feat_a1(i); 1] + (label(1) ~= i);
%   
%   score_pred_2 = zeros(1, action_n(2));
%   for j = 1:action_n(2)
%      
%     if floor((j - 1) / orient_n) + 1 ~= i
%       score_pred_2(j) = -inf;
%       continue;
%     end
%     
%     idx = action_n(1) * 2 + (i - 1) * action_n(2) * 2 + ...
%           (j - 1) * 2 + 1;
%     score_pred_2(j) = w(idx:idx+1) * [feat_a2(j); 1];
%     
%   end
%   
%   [val idx] = max(score_pred_2);
%   orient_pred(i) = mod(idx-1, action_n(1)) + 1;
%   score_pred(i) = score_pred(i) + val;
%   
% end
% 
% [val label_pred(1)] = max(score_pred);
% label_pred(2) = orient_pred(label_pred(1));

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
  nodePot(1,i) = w(idx:idx+featdims-1) * [feat_a1(1,i);feat_a1(2,i);1] ...
                 + (label(1) ~= i);
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
% % for i = 1:action_n(1)
% %   idx = (i - 1) * featdims_a(1) + 1 : i * featdims_a(1);
% %   nodePot((p_i-1)*label_n+1,i) = w(idx) * feat_a1(p_i,:)' + ...
% %                 loss_r(1) * (label(p_i,1)~=i) * action_r{1}(label(p_i,1));                             
% % end
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
%   nodePot((p_i-1)*label_n+2,i) = w(idx) * feat_a2(p_i,:)' + ...
%               loss_r(2) * (label(p_i,2)~= i) * action_r{2}(label(p_i,2));
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
%   nodePot((p_i-1)*label_n+3,i) = w(idx) * feat_o(p_i,:)'+ ...
%               + loss_r(3) * (label(p_i,3)~=i) * action_r{3}(label(p_i,3));
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
% % for i = 1:action_n(1)
% %   for j = 1:action_n(2)
% %     idx = idx_start + (i-1) * action_n(2) + j;
% %     edgePot(i,j,edge_idx) = w(idx);
% %   end
% % end
% 
% node_i = (p_i-1)*label_n+2;
% node_j = (p_i-1)*label_n+3;
% edge_idx = find(ismember(edgeStruct.edgeEnds, [node_i node_j], 'rows'));  
%     
% idx_start = idx_start + action_n(1) * action_n(2);
% for i = 1:action_n(2)
%   for j = 1:action_n(3)    
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
%     if conn(i, j) == 0
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
% % nodePot = zeros(nNodes, maxState);
% % edgePot = zeros(maxState, maxState, nEdges);
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
%   nodePot((p_i-1)*label_n+1,i) = w(idx) * feat_a1(p_i,:)' + ...
%                 loss_r(1) * (label(p_i,1)~=i) * action_r{1}(label(p_i,1));                             
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
%   nodePot((p_i-1)*label_n+2,i) = w(idx) * feat_a2(p_i,:)' + ...
%               loss_r(2) * (label(p_i,2)~= i) * action_r{2}(label(p_i,2));
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
%   nodePot((p_i-1)*label_n+3,i) = w(idx) * feat_o(p_i,:)'+ ...
%               + loss_r(3) * (label(p_i,3)~=i) * action_r{3}(label(p_i,3));
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
%   nodePot(i, :) = exp(nodePot(i,:));
% end
% 
% edgePot_ini = edgePot;

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
%   nodePot(i, :) = exp(nodePot(i,:));
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
%   %pot(inter_i) = sum(pot_tmp);
% 
%   pot(inter_i) = 1;
%   label0 = label_pred_all{inter_i};
% 
%   for i = 1:nNodes
%     pot(inter_i) = pot(inter_i) * nodePot(i,label0(i));
%   end
% 
%   for i = 1:nEdges
%     edge_idx = edgeStruct.edgeEnds(i,:);
%     pot(inter_i) = pot(inter_i) * edgePot(label0(edge_idx(1)), label0(edge_idx(2)), i);
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
% end




% score_pred = zeros(action_n(1), action_n(2), action_n(3));
% for i = 1:action_n(1)
%   for j = 1:action_n(2)
%     for k = 1:action_n(3)
%       score_pred(i,j,k) = nodePot(1,i) + nodePot(2,j) + nodePot(3,k) + ...
%                           edgePot(i,j,1) + edgePot(j,k,2);
%     end
%   end
% end
% 
% [val idx] = max(score_pred(:));
% [label_pred(1) label_pred(2) label_pred(3)] = ind2sub(action_n, idx);



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
%   idx = (i - 1) * featdims_a(1) + 1 : i * featdims_a(1);
%   idx_f = idx_f_start + (i - 1) * action_n(1) + label_f(1);
%   nodePot{1}(i) = w(idx) * feat_a1 + w(idx_f) + loss_r(1) * (label(1)~=i);
% end
%   
% % level 2
% idx_start = action_n(1) * featdims_a(1);
% idx_f_start = idx_f_start + action_n(1)^2;
% for i = 1:action_n(2)
%   idx = idx_start + (i - 1) * featdims_a(2) + 1 : ...
%         idx_start + i * featdims_a(2);
%   idx_f = idx_f_start + (i-1)*action_n(2) + label_f(2);
%   nodePot{2}(i) = w(idx) * feat_a2 + w(idx_f) + loss_r(2) * (label(2)~= i);
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
%   nodePot{3}(i) = nodePot{3}(i) + w(idx_f) + loss_r(3) * (label(3)~=i);
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
% edgePot{1,2} = gamma;
% edgePot{2,1} = gamma;
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
% edgePot{2,3} = gamma;
% edgePot{3,2} = gamma;
% 
% % for i = 1:nNodes
% %   nodePot{i} = exp(nodePot{i});
% % end
% 
% [bel, converged] = inference(conn,edgePot,nodePot,'loopy','sum_or_max',1,...
%                              'max_iter',1);
% 
% label_pred = zeros(1, nNodes);                         
% for i = 1:nNodes
%   [val, label_pred(i)] = max(bel{i});
% end

% for i = 1:nNodes
%   [val, label_pred(i)] = max(nodePot{i});
% end

% function label_pred = infer_label_f(w, label, label_f, action_n, orient_n, ...
%                        feat_a1, feat_a2, feat_a3, feat_o, action_r, loss_r)
% 
% np = size(feat_a1,1);                        
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
% % nodePot = zeros(nNodes, maxState);
% % edgePot = zeros(maxState, maxState, nEdges);
% 
% nodePot = -inf*ones(nNodes, maxState);
% edgePot = -inf*ones(maxState, maxState, nEdges);
% 
% for p_i = 1:np
%     
%   if label_f(p_i, 1) == 0
%     continue;
%   end
% 
% idx_f_start = featdims_a * action_n' + featdims_o * orient_n + ...
%               action_n(1) * action_n(2) + action_n(2) * action_n(3);
% 
% % level 1
% for i = 1:action_n(1)
%   idx_f = idx_f_start + (label(p_i,1) - 1) * action_n(1) + i;
%   nodePot((p_i-1)*label_n+1,i) = w(idx_f) + loss_r(1) * ...
%                   (label_f(p_i,1)~=i) + action_r{1}(label_f(p_i,1));
% end
%   
% % level 2
% idx_f_start = idx_f_start + action_n(1)^2;
% for i = 1:action_n(2)
%   idx_f = idx_f_start + (label(p_i,2)-1)*action_n(2) + i;
%   nodePot((p_i-1)*label_n+2,i) = w(idx_f) + loss_r(2) * ...
%                  (label_f(p_i,2)~= i) + action_r{2}(label_f(p_i,2));
% end
% 
% % level 3
% idx_f_start = idx_f_start + action_n(2)^2;
% for i = 1:action_n(3)  
%   idx_f = idx_f_start + (label(p_i,3)-1)*action_n(3) + i;
%   nodePot((p_i-1)*label_n+3,i) = w(idx_f) + loss_r(3) * ...
%                   (label_f(p_i,3)~=i) + action_r{3}(label_f(p_i,3));
% end
% 
% idx_f_start = featdims_a * action_n' + featdims_o * orient_n;
% 
% for i = 1:action_n(1)
%   for j = 1:action_n(2)
%     idx = idx_f_start + (i-1) * action_n(2) + j;
%     edgePot(i,j,(p_i-1)*2+1) = w(idx);
%   end
% end
% 
% idx_f_start = idx_f_start + action_n(1) * action_n(2);
% %gamma = zeros(action_n(2), action_n(3));
% for i = 1:action_n(2)
%   for j = 1:action_n(3)
%     action_j = floor((j - 1) / action_n(2)) + 1;
%     if i ~= action_j
%       edgePot(i,j,(p_i-1)*2+2) = -inf;
%     else
%       idx = idx_f_start + (i-1) * action_n(3) + j;
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
% [pot label_pred_all] = UGM_Decode_Tree(nodePot,edgePot,edgeStruct);
% 
% label_pred = zeros(np, label_n);
% for i = 1:np
%   if label_f(i, 1) == 0
%     continue;
%   end
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
%   idx_f = idx_f_start + (label(1) - 1) * action_n(1) + i;
%   nodePot(1,i) = w(idx_f) + loss_r(1) * (label_f(1)~=i);
% end
%   
% % level 2
% idx_f_start = idx_f_start + action_n(1)^2;
% for i = 1:action_n(2)
%   idx_f = idx_f_start + (label(2)-1)*action_n(2) + i;
%   nodePot(2,i) = w(idx_f) + loss_r(2) * (label_f(2)~= i);
% end
% 
% % level 3
% idx_f_start = idx_f_start + action_n(2)^2;
% for i = 1:action_n(3)  
%   idx_f = idx_f_start + (label(3)-1)*action_n(3) + i;
%   nodePot(3,i) = w(idx_f) + loss_r(3) * (label_f(3)~=i);
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
%   nodePot{1}(i) = w(idx_f) + loss_r(1) * (label_f(1)~=i);
% end
%   
% % level 2
% idx_f_start = idx_f_start + action_n(1)^2;
% for i = 1:action_n(2)
%   idx_f = idx_f_start + (label(2) - 1) * action_n(2) + i;
%   nodePot{2}(i) = w(idx_f) + loss_r(2) * (label_f(2)~= i);
% end
% 
% % level 3
% idx_f_start = idx_f_start + action_n(2)^2;
% for i = 1:action_n(3)
%   idx_f = idx_f_start + (label(3) - 1) * action_n(3) + i;
%   nodePot{3}(i) = w(idx_f) + loss_r(3) * (label_f(3)~=i);
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

% %label = label_1s;
% score_pred = zeros(label_n(1) * label_n(2), 1);
% featdims_a = length(feat_a);
% featdims_o = length(feat_o);
% featdims_a_o = length(feat_a_o);
% %featdims_a_o = 2;
% %label_gt_a = label(1);
% %label_gt_a_o = (label(1)-1)*label_n(2) + label(2);
% label_gt_a_o = (label(1)-1)*label_n(2) + label(2);
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
%     score_pred(k) = w(idx_a) * feat_a + (label(1) ~= label_a);
%     %score_pred(k) = w(idx_a) * feat_a;
% 
%     idx_o_start = label_n(1) * featdims_a;
%     idx_o = idx_o_start + (label_o - 1) * featdims_o + 1: ...
%             idx_o_start + label_o * featdims_o;
%     score_pred(k) = score_pred(k) + w(idx_o) * feat_o;
%  
%     idx_a_o_start = idx_o_start + label_n(2) * featdims_o;
%     idx_a_o = idx_a_o_start + (label_a_o - 1) * featdims_a_o + 1 : ...
%               idx_a_o_start + label_a_o * featdims_a_o;
%     score_pred(k) = score_pred(k) + w(idx_a_o)*feat_a_o + ...
%                     (label_gt_a_o ~= label_a_o);
%      
%     %score_pred(k) = score_pred(k) + w(idx_a_o)*feat_a_o;
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
%     
%     %score_pred(k) = score_pred(k) + w(idx_a_o)*feat_a_o;
% 
%     %score_pred(k) = score_pred(k) + (label_gt_a_o ~= label_a_o);
%    
% %    idx2_start = label_n(1) * featdims_a;
% %    idx2 = idx2_start + (label_l2 - 1) * featdims_a_o + 1 : ...
% %           idx2_start + label_l2 * featdims_a_o;       
%     %score_pred(k) = score_pred(k) + w(idx2) * feat_a_o + (label(2) ~= label_l2);
% %     score_pred(k) = score_pred(k) + w(idx2) * [feat_a_o(label_l2); 1] ...
% %                     + (label(2) ~= label_l2);
%     %score_pred(k) = score_pred(k) + w(idx2) * feat;
%   end
% end
% 
% [val k_pred] = max(score_pred);
% label_pred(1) = floor((k_pred - 1) / 5) + 1;
% label_pred(2) = mod(k_pred - 1, 5) + 1;
% %label_pred(2) = k_pred;


% function label_pred = infer_label_1s(w, label_prev, label, ...
%                                   label_n, feat_a, ...
%                                   feat_o, feat_a_o)
% 
% score_pred = zeros(label_n(1) * label_n(2), 1);
% featdims_a = length(feat_a);
% featdims_o = length(feat_o);
% featdims_a_o = length(feat_a_o);
% %featdims_a_o = 2;
% %label_gt_a = label(1);
% %label_gt_a_o = (label(1)-1)*label_n(2) + label(2);
% 
% label_gt_a_o = (label(1)-1)*label_n(2) + label(2);
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
%     score_pred(k) = score_pred(k) + w(idx_f_a) + (label_a ~= label(1));
%     %score_pred(k) = score_pred(k) + w(idx_f_a);
% 
%     % action+orientation - action+orientation@future
%     idx_f_a_o_start = idx_f_a_start + label_n(1)^2;
%     idx_f_a_o = idx_f_a_o_start + (label_prev_a_o-1)*label_n(1)^2 + label_a_o;
%     score_pred(k) = score_pred(k) + w(idx_f_a_o) + (label_gt_a_o ~= label_a_o);
% 
%   end
% end
% 
% [val k_pred] = max(score_pred);
% label_pred(1) = floor((k_pred - 1) / 5) + 1;
% label_pred(2) = mod(k_pred - 1, 5) + 1;

