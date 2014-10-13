function f = fChainedCrescentI(x)
N = numel(x);
f = 0;
for i=1:(N-1)
	f = f - x(i) + 2*(x(i)^2 + x(i+1)^2 - 1) + 1.75 * abs(x(i)^2 + x(i+1)^2 - 1);
end
f = f + 0.5*1*sum((x-0).^2);

