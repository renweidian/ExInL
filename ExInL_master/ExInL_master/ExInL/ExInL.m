function  [Z] =  CNN_Subpace_FUS_local_lowrank( HSI, MSI,R,FBm,sf,S,para,gama,total_patches,lambda2)
% gama=1;
global sigmas
load('D:\ÎÒµÄ´úÂë\CNN_subspace\CNN_subspace\FFDNet-master\FFDNet-master\models\FFDNet_gray.mat');
%   load('G:\HSI-superresolution\FFDNet-master\FFDNet-master\models\FFDNet_Clip_gray.mat');
sig=para.sig;
% net = vl_simplenn_tidy(net);
net=vl_simplenn_move(net, 'gpu') ;
p=para.p;
mu=1e-3;


HSI3=Unfold(HSI,size(HSI),3);
% [w Rw] = estNoise(HSI3,'additive');
% [~, D]=hysime(HSI3,w,Rw);
[D,~,~]=svds(HSI3,p);
D=D(:,1:p);

RD=R*D;
L1=size(D,2);
nr=size(S,1);
nc=size(S,2);



L=size(HSI,3);



HSI_int=zeros(nr,nc,L);
HSI_int(1:sf:end,1:sf:end,:)=HSI;
FBmC  = conj(FBm);

       FBCs=repmat(FBmC,[1 1 L]);

HHH=ifft2((fft2(HSI_int).*FBCs));
  HHH1=hyperConvert2D(HHH);




%% iteration

MSI3=Unfold(MSI,size(MSI),3);

n_dr=nr/sf;
n_dc=nc/sf;


V2=zeros(p,size(MSI,1)*size(MSI,2));
G2=zeros(size(V2));
V3=V2;
G3=G2;
CCC=(gama*RD'*MSI3+D'*HHH1);


Y1=normalize(MSI3');
B=sparsepca(Y1);
Y1=Y1*B;
Y1=reshape(Y1,nr,nc);
[lables,~]=suppixel(Y1,total_patches);
 


for i=1:15
     C1=gama*(RD)'*RD+2*mu*eye(size(D,2)); 
    HR_HSI3=mu*(V2+G2/(2*mu))+mu*(V3+G3/(2*mu));
C3=CCC+HR_HSI3;


   [A] = Sylvester(C1,FBm, sf,n_dr,n_dc,C3);  
    
% Zt=hyperConvert3D(D*A,nr, nc );
%    psnr(i)=csnr(double(im2uint8(S)),double(im2uint8(Zt)),0,0)
B2=A-G2/(2*mu);
 B2=hyperConvert3D(B2,nr, nc );
 V2=zeros(size(B2));
 
sig=para.sig;
%% CNN denoiser
for jj=1:size(B2,3)
    eigen_im=(  B2(:,:,jj));
    min_x = min(eigen_im(:));
    max_x = max(eigen_im(:));
    eigen_im = eigen_im - min_x;
    scale = max_x-min_x;
    eigen_im =single (eigen_im/scale);
     input = gpuArray(eigen_im);
%      sig=sig*1.1;
    sigmas=sig/2/scale/mu;
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    BB = gather(res(end).x);
    V2(:,:,jj)=double(BB)*scale + min_x;
end



for ll=0:1:total_patches-1
      index=find(lables==ll);
      ggg=A-G3;
    temp=ggg(:,index);
    C=lambda2/(2*mu);
%      aa5= prox_nuclear (temp,C);
  
 aa3  =   repmat(mean( temp, 2 ),1,size(temp,2));
            aa4    =   temp - aa3;    
   aa5= prox_nuclear (aa4,C)+ aa3;
 V3(:,index)=aa5;
 end


V2=hyperConvert2D(V2);
G2=G2+2*mu*(V2-A);
G3=G3+2*mu*(V3-A);

mu=mu*para.gama;
end
Z=hyperConvert3D(D*A,nr, nc );