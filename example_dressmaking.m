clearvars; close all; clc;

%% DAFAULT EXAMPLE
dressmaker;

%% ALL IMAGES
close all; clc;
load('images.mat');

% TIE
[TIE2, DRESS2] = dressmaker(TIE.rgb, TIE.mask, DRESS.rgb, DRESS.mask);

% JACKET
JACKET2 = dressmaker(JACKET.rgb, JACKET.mask, DRESS.rgb, DRESS.mask);

% EGG
EGG2 = dressmaker(EGG.rgb, EGG.mask, DRESS.rgb, DRESS.mask);

% FISH
FISH2 = dressmaker(FISH.rgb, FISH.mask, DRESS.rgb, DRESS.mask);

% PEEP
PEEP2 = dressmaker(PEEP.rgb, PEEP.mask, DRESS.rgb, DRESS.mask);
