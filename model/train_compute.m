function  [fvalue gradient nothing] = train_compute(x0, auxdata)
%compute sub-gradient

action_n = auxdata.action_n;
orient_n = auxdata.orient_n;
action_fine_n = auxdata.action_fine_n;
totaldims = auxdata.totaldims;
ratio = 1;
%action_r = auxdata.action_r; % balance the loss for different categories


gradient = zeros(totaldims, 1);
fvalue = 0;

clip_idx = auxdata(1).clip.idx;
video_n = length(clip_idx);

disp('computing sub-gradient');
tic

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
  
  label = auxdata(clip_idx{i}(2)).label
  
  % use a temporal distance dependent loss function
  %t = label(4);
  %t(t < -20) = -20;
  %t(t > 0) = 0;
  %ratio = (20 + t) / 20;
  
  phi_positive = construct_phi(label, action_n, orient_n, ...
                               feat_a1, feat_a2, feat_a3, totaldims);
  fvalue_positive = x0 * phi_positive;

  label_pred = infer_label(x0, label, action_n, orient_n, action_fine_n, ...
                           feat_a1, feat_a2, feat_a3, ratio);
       
  phi_negative = construct_phi(label_pred, action_n, orient_n, ...
                               feat_a1, feat_a2, feat_a3, totaldims);

  fvalue_negative = x0 * phi_negative + ratio * (label(1) ~= label_pred(1));

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

y1 = label(1);
y2 = (y1 - 1) * orient_n + label(2);
y3 = label(3);

% level 1
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


function label_pred = infer_label(w, label, action_n, orient_n, action_fine_n, ...
                                  feat_a1, feat_a2, feat_a3, ratio)
% inference (BP)    

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
                 + ratio * (label(1) ~= i);
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

