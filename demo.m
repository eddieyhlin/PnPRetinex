clc; clear all; close all;

img = double(imread('data/lg-image3.jpg'));
img = img/255;

param.alpha = 0.3;
param.beta = 0.1; 
param.phi = 0.01;
param.delta = 0.01; 
param.ro = 1.5;
param.lpnorm = 0.4;
param.epsilon = 1e-3;
param.u = 1;
param.max_itr = 50;

gamma = 2.2;
hsv = rgb2hsv(img);
[L, R] = PnPRetinex(hsv(:,:,3), param);   
L = min(max(L,0),max(max(L)));
R = min(max(R,0),max(max(R)));
hsv(:,:,3) = R;
R_rgb = hsv2rgb(hsv);
hsv(:,:,3) = R.*(L.^(1/gamma));

outputDir='output/'; 
if ~exist(outputDir, 'dir')
    mkdir(outputDir)
end   
imwrite(hsv2rgb(hsv),[outputDir 'Result_enhanced.png']); 
imwrite(L,[outputDir 'Result_L.png']); 
imwrite(R_rgb,[outputDir 'Result_R.png']); 
