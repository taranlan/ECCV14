function [w fbest numite fstart ] = evalNRBM(varargin)
%function [w fbest numite fstart ] = evalNRBM(varargin)
% ex. evalNRBM('N',2);
% default options
%	option = struct('lambda',1,... % regularization parameter
%                	'maxiter',500,... % max number of iteration
%	                'maxCP',100,... % max number of cutting plane
%        	        'EPS',0.001,... % stop criteria gap=3%
%			'task','Chained2',... % task
%			'N',1000,...
%			'LS',0);
	option = struct('lambda',1,... % regularization parameter
                	'maxiter',500,... % max number of iteration
	                'maxCP',100,... % max number of cutting plane
        	        'EPS',0.001,... % stop criteria gap=3%
			'task','Chained2',... % task
			'N',1000,...
			'LS',0);

	if mod(numel(varargin),2)==1
        	disp('evalNCO.m : numel of varargin must be pair!!!!!');
	        return;
	end

	% get option
	for i=1:floor(numel(varargin)/2)
		str = varargin{(i-1)*2+1};
		val = varargin{i*2};
		if isstr(str)
			option = setfield(option,str,val);
		end
	end

        optionsNCBundle = struct('epsilon',option.EPS,...
                'maxiter',option.maxiter,...
                'maxCP',option.maxCP,...
                'fpositive',0,...
                'computeGapQP',1,...
                'updateDiary',1,...
                'verbosity',2,...
                'cuttingPlanAtOptimumDual',0,...
		'LS',option.LS);

	dimW = option.N;
	N = dimW;
	reg = ones(1,dimW);
%	wreg = ones(1,dimW);
	wreg = zeros(1,dimW);
	auxdata = {};

        handleF1 = '';
        handleF2 = '';
	if strcmpi(option.task,'Chained2')
		handleFGrad = @callbackChained2_FGrad;
		w0 = zeros(1,N);
		for I=1:N if(mod(I,2) >= 1) w0(I) = -1.5;else w0(I) = 2.0;end;end;
		wreg = w0;
	elseif strcmpi(option.task,'ChainedMifflin2')
                handleFGrad = @callbackChainedMifflin2_FGrad;
	        w0 = -ones(1,N);
		wreg = w0;
	elseif strcmpi(option.task,'Spike')
                handleFGrad = @callbackSpike_FGrad;
		wreg = zeros(1,N);wreg(1) = 1;
		w0 = zeros(1,N);w0(1) = 1;
	elseif strcmpi(option.task,'Water')
                handleFGrad = @callbackWater_FGrad;
                eps = 0.0001;
		wreg = ones(1,N) + rand(1,N)*eps*2 - eps ;
		w0 = wreg;
	elseif strcmpi(option.task,'Poly')
                handleFGrad = @callbackPoly_FGrad;
		wreg = randn(1,N)*0.01;
		w0 = wreg;
		Deg = 7; L = -1; U = -0.0;
		a = rand(Deg,N) * (U-L) + L;a(Deg,:)=1;
%		a = repmat(a,1,N);
		auxdata = { Deg a };
	end
	[w,fbest,numite,fstart] = GNRBM(w0,reg,option.lambda,optionsNCBundle,handleFGrad,auxdata,handleF1,handleF2,wreg);
	if (fstart==fbest)
%		keyboard;
	end


function [Remp Grad nothing] = callbackChained2_FGrad(X,auxdata)
	N = numel(X);
	Remp = 0;
	Grad = zeros(1,N);
	for I=1:(N-1)
		TEMP2 =  X(I)^2 + (X(I+1)-1.0)^2 + X(I+1) - 1;
		TEMP3 = - X(I)^2 - (X(I+1)-1.0)^2 + X(I+1) + 1.0;
		if (TEMP2 >= TEMP3)
			Remp = Remp + TEMP2;
			Grad(I) = Grad(I)+2*X(I);
			Grad(I+1) = 2*(X(I+1)-1) + 1;
		else
			Remp = Remp + TEMP3;
			Grad(I) = Grad(I) - 2*X(I);
                        Grad(I+1) = -2*(X(I+1)-1) + 1;
		end
	end
	nothing = [];
  

%======================================
function  [Remp Grad nothing]  = callbackChainedMifflin2_FGrad(x,auxdata)
	N = numel(x);
	Remp = 0;
	Grad = zeros(1,N);
	for i=1:(N-1)
        	Remp = Remp - x(i) + 2*(x(i)^2 + x(i+1)^2 - 1) + 1.75 * abs(x(i)^2 + x(i+1)^2 - 1);
		if x(i)^2 + x(i+1)^2 - 1 > 0
			Grad(i) = Grad(i) - 1 + 4*x(i) + 1.75*2*x(i);
			Grad(i+1) = 4*x(i+1) + 1.75*2*x(i+1);
		else
			Grad(i) = Grad(i) - 1 + 4*x(i) - 1.75*2*x(i);
			Grad(i+1) = 4*x(i+1) - 1.75*2*x(i+1);
		end
	end
	nothing = [];

%===========================================================================
function  [Remp Grad nothing]  = callbackSpike_FGrad(x,auxdata)
	N = numel(x);
	x2 = x*x';
	Remp = x2^0.25;
	if x2>0
		Grad = 0.5*x2^(-0.75) * x;
	else
		Grad = zeros(1,N);
	end
	nothing = [];

function  w0  = callbackSpike_Init(N)

%===========================================================================
function  [Remp Grad nothing]  = callbackWater_FGrad(x,auxdata)
	N = numel(x);
	listNeg = find(x<=0);
	listPos = find(x>0);
	fi = zeros(1,N);
	fi(listNeg) = x(listNeg).^2;
	ni = floor(log(x(listPos)) / log(2)) + 1;
	fi(listPos) = -2.^(-ni+1) .* ((x(listPos)-2.^ni).^2) + 2.^(ni);
	Remp = sum(fi);
	Grad = zeros(1,N);
	Grad(listNeg) = 2*x(listNeg);
	Grad(listPos) = -2.^(-ni+2) .* (x(listPos) - 2.^ni);
	nothing = [];

function  w0  = callbackWater_Init(N)
	w0 = ones(1,N) + randn(1,N)*0.001;

%===========================================================================
function  [Remp Grad nothing]  = callbackPoly_FGrad(x,auxdata)
	[Deg a] = deal(auxdata{:});
	N = numel(x);
	pi = zeros(1,N);
	xp = abs(x);
	xd = ones(1,N);
	gi = zeros(1,N);
	for d=1:Deg
		xd_old = xd;
		xd = xd .* xp;
		pi = pi + xd .* a(d,:);
		gi = gi + (xd_old * d) .* a(d,:);
	end
	Remp = sum(pi);
	Grad = sign(x) .* gi;
	nothing = [];

