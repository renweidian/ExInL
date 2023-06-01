% This is the testing demo of FFDNet for denoising noisy grayscale images corrupted by
% AWGN with clipping setting. The noisy input is 8-bit quantized.
% "FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising"
%  2018/03/23
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)

%clear; clc;
format compact;
global sigmas; % input noise level or input noise level map
addpath(fullfile('utilities'));

folderModel = 'models';
folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'BSD68','Set12'}; % testing datasets
setTestCur  = imageSets{1};      % current testing dataset

showResult  = 1;
useGPU      = 1; % CPU or GPU. For single-threaded (ST) CPU computation, use "matlab -singleCompThread" to start matlab.
pauseTime   = 0;

imageNoiseSigma = 25;  % image noise level, 25.5 is the default setting of imnoise( ,'gaussian')
inputNoiseSigma = 25;  % input noise level

folderResultCur       =  fullfile(folderResult, [setTestCur,'_Clip_',num2str(imageNoiseSigma),'_',num2str(inputNoiseSigma)]);
if ~isdir(folderResultCur)
    mkdir(folderResultCur)
end

load(fullfile('models','FFDNet_Clip_gray.mat'));
net = vl_simplenn_tidy(net);

% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end

if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,setTestCur,ext{i})));
end

% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));

for i = 1:length(filePaths)
    
    % read images
    label = imread(fullfile(folderTest,setTestCur,filePaths(i).name));
    [w,h,~]=size(label);
    if size(label,3)==3
        label = rgb2gray(label);
    end
    
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    label = im2single(label);
    
    % add noise
    randn('seed',0);
    %input = imnoise(label,'gaussian'); % corresponds to imageNoiseSigma = 25.5;
    input = imnoise(label,'gaussian',0,(imageNoiseSigma/255)^2);
    
    if mod(w,2)==1
        input = cat(1,input, input(end,:)) ;
    end
    if mod(h,2)==1
        input = cat(2,input, input(:,end)) ;
    end
    
    % tic;
    if useGPU
        input = gpuArray(input);
    end
    
    % set noise level map
    sigmas = inputNoiseSigma/255; % see "vl_simplenn.m".
    
    % denoising
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    output = res(end).x;
    
    
    if mod(w,2)==1
        output = output(1:end-1,:);
        input  = input(1:end-1,:);
    end
    if mod(h,2)==1
        output = output(:,1:end-1);
        input  = input(:,1:end-1);
    end
    
    if useGPU
        output = gather(output);
        input  = gather(input);
    end
    % toc;
    % calculate PSNR, SSIM and save results
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
    if showResult
        imshow(cat(2,im2uint8(input),im2uint8(label),im2uint8(output)));
        title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        %imwrite(im2uint8(output), fullfile(folderResultCur, [nameCur, '_' num2str(imageNoiseSigma,'%02d'),'_' num2str(inputNoiseSigma,'%02d'),'_PSNR_',num2str(PSNRCur*100,'%4.0f'), extCur] ));
        drawnow;
        pause(pauseTime)
    end
    disp([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
    PSNRs(i) = PSNRCur;
    SSIMs(i) = SSIMCur;
end

disp([mean(PSNRs),mean(SSIMs)]);




