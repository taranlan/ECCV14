p = [-20 -15 -10 -5  0 5];
len = [1 3 5 7 9];

for m = 1:length(len)
    
  for i = 1:length(p)-1
      
  i_start = p(i);
  if i_start < 0 
    i_end = 0;
  else
    i_end = 5;
  end
    
  main_test([i_start i_end], len(m));
    
  end
end