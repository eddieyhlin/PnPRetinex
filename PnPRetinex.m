function [L, R, Mh] = PnPRetinex(I, para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('GF'));

% set parameters
alpha = para.alpha;
beta = para.beta;
phi = para.phi;
delta = para.delta;
ro = para.ro;
lpnorm = para.lpnorm;
epsilon = para.epsilon;
u = para.u;
max_itr = para.max_itr;

%initialize variables
[row,col,ch]= size(I);    %dimension of the image
dim         = [row,col,ch];
PNum        = dim(1)*dim(2);    %number of pixels in the image

L       = I;
R       = zeros(dim);
B       = L;

Th          = zeros(dim);
Zh          = zeros(dim);
Tv          = zeros(dim);
Zv          = zeros(dim);
S           = L;
Y           = zeros(dim);
residualR   = inf;
residualL   = inf;

x_corrd = reshape( repmat( [1:PNum], 2,1 ), 1, [] );
x_corrd = x_corrd';
%diagonal matrix for A horizontal
y_corrd = zeros(PNum*2,1);
value = zeros(PNum*2,1);
for i = 1:PNum
    y_corrd(i*2-1,1) = i;
    y_corrd(i*2,1)   = mod(dim(1)*(dim(2)-1) + i, PNum);
    if (mod(dim(1)*(dim(2)-1) + i, PNum) == 0)
        y_corrd(i*2,1) = PNum;        
    end
    if (i <= dim(1))
        value(i*2-1,1) = 0; %boundary
        value(i*2,1) = 0; %boundary
    else
        value(i*2-1,1) = 1;
        value(i*2,1) = -1;
    end
end
Dh = sparse(x_corrd,y_corrd,value,PNum,PNum);
Dh_t = Dh';

%diagonal matrix for A vertical
y_corrd = zeros(PNum*2,1);
value = zeros(PNum*2,1);
for i = 1:PNum
    y_corrd(i*2-1,1) = i;
    if (mod(i,dim(1)) == 1)
        y_corrd(i*2,1) = i + dim(1) - 1;  
        value(i*2-1,1) = 0; %boundary
        value(i*2,1) = 0; %boundary
    else
        y_corrd(+ i*2,1) = i - 1;
        value(i*2-1,1) = 1;
        value(i*2,1) = -1;
    end
end

Dv = sparse(x_corrd,y_corrd,value,PNum,PNum);
Dv_t = Dv';

DhDhpDvDv = Dh_t*Dh + Dv_t*Dv;

%Weight M for R
[Mh,Mv] = R_weight_method(I, Dh, Dv);

DMMDh = Dh_t*diag(sparse(Mh(:)))*Dh;
DMMDv = Dv_t*diag(sparse(Mv(:)))*Dv;
deltaDMMD = delta*(DMMDh + DMMDv)/2;

% main loop
itr = 1;
while((residualR>epsilon && residualL>epsilon) && itr<=max_itr)
    %store x, v, u from previous iteration for psnr residual calculation
    
    R_old = R;
    L_old = L;
    
    %update R
    lhs = diag(sparse(L(:).*L(:) + 0.000001)) + deltaDMMD;
    rhs = L(:).*I(:);
    C = ichol( lhs, struct( 'michol', 'on' ) );
    [R,~] = pcg( lhs, rhs, 1e-4, 100, C, C' );
    R = reshape(R,dim);

    %update L
    lhs = diag(sparse((2*R(:).*R(:))+(2*alpha+u))) + u*DhDhpDvDv;
    TZh = Th - Zh/u;
    TZv = Tv - Zv/u;
    uDt_z = u*(Dh_t*(TZh(:)) + Dv_t*(TZv(:)));
    rhs = 2*R(:).*I(:) + uDt_z + u*(S(:) - Y(:)/u) + 2*alpha*B(:);
    C = ichol( lhs, struct( 'michol', 'on' ) );
    [L,~] = pcg( lhs, rhs, 1e-4, 100, C, C' );
    L = reshape(L,dim);  
    
    %shrinkage Lp norm
    e     = (beta/u).^(2-lpnorm);
    eps      = 0.00001;
    
    dLh = Dh*(L(:));
    dLh = reshape(dLh,dim);
    th     = dLh + (Zh/u);
    ths    = abs(th)-(e*((abs(th) + eps).^(lpnorm-1)));
    thres  = ths > 0;
    Th     = sign(th).*(ths.*thres);
    
    dLv = Dv*(L(:));
    dLv = reshape(dLv,dim);    
    tv     = dLv + (Zv/u);
    tvs    = abs(tv)-(e*((abs(tv) + eps).^(lpnorm-1)));
    thres  = tvs > 0;
    Tv     = sign(tv).*(tvs.*thres);
        
    %update S     
    Sin     = L + Y/u;
    phiu    = phi / (u^0.5);
    S1      = guidedfilter(Sin,Sin,5,phiu);
    %S1      = imguidedfilter(Sin,Sin,'NeighborhoodSize',[11 11],'DegreeOfSmoothing',phiu);
    
    phiu     = 0.187 / (u^0.5);    
    Sin2     = max(Sin,0.0001);
    Sin2     = log(Sin2);
    S2       = guidedfilter(Sin2,Sin2,5,phiu);
    %S2       = imguidedfilter(Sin2,Sin2,'NeighborhoodSize',[11 11],'DegreeOfSmoothing',phiu);
    S2       = exp(S2);
    
    thresS   = 0.5^2.2;
    interval = 10/256;
    S_w     = min(1,(max(0,Sin-thresS)/interval));
    S        = S1.*(S_w) + S2.*(1-S_w);
    
    %update langrangian multiplier Z
    Y      = Y + u*(L-S);
    Zh     = Zh + u*(dLh-Th);
    Zv     = Zv + u*(dLv-Tv);
    
    %update u
    u = u*ro;
    
    %calculate residual
    residualR = (1/sqrt(PNum))*(sqrt(sum(sum(sum((R-R_old).^2)))));
    residualL = (1/sqrt(PNum))*(sqrt(sum(sum(sum((L-L_old).^2)))));

    fprintf('Iteration%3g: Diff_L: %2.4e, Diff_R: %2.4e\n', itr, residualL, residualR);
    
    itr = itr+1;
end
end

function [RWh RWv] = R_weight_method(I, Dh, Dv)

[row,col,ch]= size(I);    %dimension of the image
dim         = [row,col,ch];

filt = fspecial('gaussian',5,2);    
EI1 = imfilter((I.^(1/2.2)).^2,filt);
EI2 = imfilter((I.^(1/2.2)),filt).^2;
A = abs(EI1-EI2);
A = min(A*4500,1);    
A = 1-A;  
   
%%%%%%%%%%      
Id = I.^(1/2.2);
filt = fspecial('gaussian',5,2);
    
dIh = Dh*Id(:);
dIh = reshape(dIh,dim);         
avgRh = imfilter(dIh,filt);    
avgRh = abs(avgRh);    
B = min(avgRh*220, 1);  % texture map
B(:,1) = 0;
RWh = B;    
    
dIv = Dv*Id(:);
dIv = reshape(dIv,dim);        
avgRv = imfilter(dIv,filt);
avgRv = abs(avgRv);
B = min(avgRv*220, 1);
B(1,:) = 0;    
RWv = B;  
        
R = RWv .* (RWv > RWh) + RWh .* (RWv <= RWh);
R = 1-R;
    
r = 2.5;
C = A.*R;
C = C.^r;
         
RWv = C;
RWh = C;

end