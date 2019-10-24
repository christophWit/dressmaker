function [NEWout, DRESSout, FIG] = dressmaker(NEW0, Nmask, DRESS, Dmask)
% 2019.10.22 * [cw]

%% SETTINGS

if nargin < 4
    Dmask = imread('dress_mask.png');
    if nargin < 3
        DRESS = imread('dress.png');
        if nargin < 2
            Nmask = imread('tie_mask.bmp');
            if nargin < 1
               NEW0 = imread('tie.jpg');
clc; close all;
end; end; end; end

SETTINGS.mon_xyY    = [];
SETTINGS.mon_ldt    = [];
SETTINGS.mapdim     = [2,3]; % Default: Only chromatic dimensions (2 and 3)
SETTINGS.swaphue    = []; % if empty, determined automatically.
SETTINGS.bg         = [100 0 0]; % 100 0 0 = white
SETTINGS.plot       = 'all';
SETTINGS.design     = 'default';

DESIGN     = struct(...
    'addplots',     {{'all'...
                        }},...
    'sig_symbols', {{'°', '*', '**', '***'}},... {'', '*', '\bullet', ''};
    'font', 'Arial',...
    'fontsize',  10,...
    'cn_xtx', 'num',... % or colour name pairs --> allpairs or minmax
    'linewidth', 1,...
    'fatlinewidth', 2,...
    'markersize', 5,...
    'vplabels', 'sexed'); % original / sexed

%% DEFAULT MONITOR SPECS
[srgb_xyY, srgb_ldt] = srgb; % To convert original images
if isempty(SETTINGS.mon_xyY)
    SETTINGS.mon_xyY = srgb_xyY;
    SETTINGS.mon_ldt = srgb_ldt;
    fprintf('WARNING: Using default sRGB calibration!\n');
end

FIG = [];

%% CONVERT ORIGINAL IMAGES
wp_XYZ = rgb2XYZ([255 255 255],SETTINGS.mon_xyY, 2);
NEW0 = double(NEW0);
DRESS = double(DRESS);

DRESS2 = RGB2Luv(DRESS, wp_XYZ, SETTINGS.mon_xyY, SETTINGS.mon_ldt);
NEW2 = RGB2Luv(NEW0, wp_XYZ, SETTINGS.mon_xyY, SETTINGS.mon_ldt);

% APPLY MASKS -------------------------------------------------------------
Dmask = logical(Dmask(:,:,1));
Nmask = logical(Nmask(:,:,1));

D_Lxy2D = mask_2Dmapper(DRESS2, Dmask);
N_Lxy2D = mask_2Dmapper(NEW2, Nmask);

%% ONE HUE
[NEWout.twoD, N_onehue_std, DRESSout.twoD, N_onehue, h] = mod_1huemapper(D_Lxy2D, Dmask, N_Lxy2D, Nmask, SETTINGS, DESIGN);
FIG = [FIG; h];

%% MAPPING
NEWout.threeD = mask_3Dmapper(NEWout.twoD, Nmask, SETTINGS.bg);
NEWout.rgb = Luv2RGB(NEWout.threeD, wp_XYZ, SETTINGS.mon_xyY, SETTINGS.mon_ldt);

DRESSout.threeD = mask_3Dmapper(DRESSout.twoD, Dmask, SETTINGS.bg);
DRESSout.rgb = Luv2RGB(DRESSout.threeD, wp_XYZ, SETTINGS.mon_xyY, SETTINGS.mon_ldt);

%% ****************************** SUBMODULES ******************************

%% mod_1huemapper
function [N_mapped2onehue, N_onehue_std, D_onehue, N_onehue, fig_h] = mod_1huemapper(D_Lxy2D, Dmask, N_Lxy2D, Nmask, SETTINGS, DESIGN)
% 2018.11.20 * [cw]

if nargin < 5
    SETTINGS.mapdim = [2 3];
    SETTINGS.plot = 'all';
end

% ONE HUE -----------------------------------------------------------------
[D_onehue, D_pc, D_expl] = hue_aligner(D_Lxy2D(:,SETTINGS.mapdim));
if numel(SETTINGS.mapdim) == 2
    D_onehue = [D_Lxy2D(:,1), D_onehue];
end
fprintf('Explained variance: %.2f%%\n', D_expl(1));

[N_onehue,N_pc,N_expl] = hue_aligner(N_Lxy2D(:,SETTINGS.mapdim));
if numel(SETTINGS.mapdim) == 2
    N_onehue = [N_Lxy2D(:,1), N_onehue];
end

N_swaphue = N_onehue;
if SETTINGS.swaphue
    N_swaphue(:,SETTINGS.mapdim) = -N_onehue(:,SETTINGS.mapdim);
end
N_onehue_std = std_mapper(D_onehue, N_swaphue);
fprintf('Explained variance: %.2f%%\n', N_expl(1));

% MAP ON DRESS PC ---------------------------------------------------------
x = N_swaphue(:,SETTINGS.mapdim(1));
M = mean(x);
N_mapped2onehue = (x-M)*D_pc';
if numel(SETTINGS.mapdim) == 2
    N_mapped2onehue = [N_swaphue(:,1), N_mapped2onehue];
end
N_mapped2onehue = std_mapper(D_onehue, N_mapped2onehue);

switch lower(SETTINGS.plot)
    case {'illu', 'all'}
        fig_h = figure('Name', 'Algo', 'NumberTitle', 'off');
        
        % DRESS -----------------------------------------------------------
        % ORIGNAL DRESS:
        subplot(3,3,1)
        imshow2D(D_Lxy2D, Dmask, SETTINGS.bg);
        
        % DRESS DISTRIBUTIONS:
        subplot(3,3,4)
        hold on
        image_scatter(D_Lxy2D, 2, 30)
        image_scatter(D_onehue, 2, 30)
        plot(D_onehue(:,2),D_onehue(:,3),'k-', 'LineWidth', 2);
        plot(0, 0, 'ko', 'MarkerFaceColor', 'w');
        hold off
        xlabel('Green-red [u*]');
        ylabel('Blue-yellow [v*]');
        title('One Hue Dress', 'FontWeight', 'bold');
        axis equal;
        
        % ONE HUE DRESS:
        subplot(3,3,7)
        imshow2D(D_onehue, Dmask, SETTINGS.bg);
        
        % NEW -------------------------------------------------------------
        % ORIGINAL NEW:
        subplot(3,3,2)
        imshow2D(N_Lxy2D, Nmask, SETTINGS.bg);

        % DISTRIBUTION OF NEW IMAGE:
        subplot(3,3,5)
        hold on
        image_scatter(N_Lxy2D, 2, 30)
        image_scatter(N_onehue, 2, 30)
        plot(N_onehue(:,2),N_onehue(:,3),'k-', 'LineWidth', 2);
        plot(0, 0, 'ko', 'MarkerFaceColor', 'w');
        hold off
        set(gca, 'FontName', DESIGN.font, 'FontSize', DESIGN.fontsize');
        xlabel('Green-red [u*]');
        title('One Hue New', 'FontWeight', 'bold');
        axis equal;
        
        % ONE-HUE NEW:
        subplot(3,3,8)
        imshow2D(N_onehue, Nmask, SETTINGS.bg);

        % MAPPED NEW ------------------------------------------------------
        subplot(3,3,3)
        imshow2D(N_swaphue, Nmask, SETTINGS.bg);
        
        % DISTRIBUTION OF MAPPED:
        subplot(3,3,6)
        hold on
        image_scatter(N_swaphue, 2, 30);
        image_scatter(N_mapped2onehue, 2, 30);
        plot(N_onehue_std(:,2),N_onehue_std(:,3),'k:', 'LineWidth', 1);
        plot(0, 0, 'ko', 'MarkerFaceColor', 'w');
        hold off
        xlabel('Green-red [u*]');
        title('One Hue + STD mapped ', 'FontWeight', 'bold');
        axis equal;
        
        % FINAL MAPPED:
        subplot(3,3,9)
        imshow2D(N_mapped2onehue, Nmask, SETTINGS.bg);
end

%% ***************************** SUBFUNCTIONS *****************************

%% image_scatter
function image_scatter(Lxy, dim, msz, edge)

if nargin < 4
    edge = 1;
    if nargin < 3
        msz = 30;
    end
end

[mon_xyY, mon_ldt] = srgb;
wp_XYZ = rgb2XYZ([1 1 1],mon_xyY, 2);

Lxy2 = unique(Lxy, 'rows');
rgb = Luv2RGB(Lxy2, wp_XYZ, mon_xyY, mon_ldt, dim);

if edge
    scatter(Lxy2(:,2), Lxy2(:,3), msz, 'ko');
end
scatter(Lxy2(:,2), Lxy2(:,3), msz, rgb/255, 'filled');

%% imshow2D
function imshow2D(Lxy2D, mask, bg, wp_XYZ, mon_xyY, mon_ldt)

if nargin < 4
    [mon_xyY, mon_ldt] = srgb;
    wp_XYZ = rgb2XYZ([1 1 1],mon_xyY, 2);
end

Lxy3D = mask_3Dmapper(Lxy2D, mask, bg);
RGB = Luv2RGB(Lxy3D, wp_XYZ, mon_xyY, mon_ldt,3);
imshow(RGB/255);

%% std_mapper
function NEW2d = std_mapper(D2d, N2d)
% 2018.12.14 * [cw]

% SET MEAN AND STD TO DRESS -----------------------------------------------
dM = mean(D2d);
dS = std(D2d);
nM = mean(N2d);
nS = std(N2d);
for k = 1:3
    zN(:,k) = (N2d(:,k) - nM(:,k))/nS(:,k);
    NEW2d(:,k) = (zN(:,k)*dS(:,k)) + dM(:,k);
end

%% gammacorrector
function RGBg = gammacorrector(RGBl, LDT, dim)
% 2012aug30 [cw]

if nargin < 3
    dim = 3;
end

out1 = sum(sum(sum(RGBl>LDT(end,1)+0.5,3),2),1); % usually 255
out2 = sum(sum(sum(RGBl<-0.5,3),2),1); % usually 255
out = out1+out2;
n = size(RGBl>LDT(end,1),3) * size(RGBl>LDT(end,1),2) * size(RGBl>LDT(end,1),1); 
if out > 0
    fprintf('WARNING: %d / %.0f%% are out of gamut, %d / %.0f%% >max and %d / %.0f%%\n', out, out/n*100, out1, out1/n*100, out2, out2/n*100);
end

if dim == 2
     for dm = 1:3
        RGBg(:,dm) = gammacorrector_subfunction2(RGBl(:,dm), LDT(:,dm));
     end
elseif dim == 3
    for dm = 1:3
        RGBg(:,:,dm) = gammacorrector_subfunction2(RGBl(:,:,dm), LDT(:,dm));
    end
end

%% gammainvcorrector
function [RGB] = gammainvcorrector(RGB0, mon_ldt, dim, mon_rgbmax)
% 2012.12.15 [cw].

% COPE WITH INPUT VARIATIONS
if nargin < 3
    dim = ndims(RGB0);
end
% CHECK RGB0:
if size(RGB0, dim) ~= 3
    if size(RGB0, dim) == 4
        warning('Input format: RGB0 has 4 entries; 4th dimensions considered as alpha (transparency)');
    else
        error('Input format: RGB0 does not have 3 entries for R, G, and B');
    end
end

% CHECK mon_ldts:
if size(mon_ldt,2) ~= 3
    error('Input format: mon_ldt do not have 3 columns for R, G, and B');
end

% GAMMA CORRECTION = LOOK UP
for dm = 1:3
    if dim == 2
        RGB(:,dm)   = gammainvcorrector_subfunction(RGB0(:,dm),mon_ldt(:,dm),mon_rgbmax);
    elseif dim == 3
        RGB(:,:,dm) = gammainvcorrector_subfunction(RGB0(:,:,dm),mon_ldt(:,dm),mon_rgbmax);
    end
end

% gammainvcorrector_subfunction
function RGB = gammainvcorrector_subfunction(RGB0, mon_ldt, mon_rgbmax)
% For R,G,B separately (!).

RGB0 = overflow_checker(RGB0,mon_rgbmax);
sz = size(RGB0);
inds = ~isnan(RGB0);
inds_nan = isnan(RGB0);
RGB = mon_ldt(RGB0(inds) + 1);
RGB(inds_nan) = NaN;
RGB = reshape(RGB, sz(1), sz(2));

% overflow_checker
function RGB = overflow_checker(RGB0, mon_rgbmax)
RGB = RGB0;
inds = RGB > mon_rgbmax;
inds2 = RGB < 0;
RGB(inds) = mon_rgbmax;
RGB(inds2) = 0;
if sum(inds) > 0 | sum(inds2) > 0
    disp('overflows truncated to the range [0 255]')
end

%% gammacorrector_subfunction2
function RGBg1 = gammacorrector_subfunction2(RGBl1, LDT1)
y1 = (1:numel(LDT1))'-1;
RGBg1 = round(interp1(LDT1, y1, RGBl1, 'linear', 'extrap'));

%% gammacorrection
function [R_corrected, G_corrected, B_corrected] = gammacorrection(R, G, B, LUT, LUT2, LUT3)
% 2009 [cw].

% COPE WITH INPUT VARIATIONS
if nargin == 4 && iscell(LUT)
    redlut      = LUT{1};
    greenlut    = LUT{2};
    bluelut     = LUT{3};
elseif nargin > 4 && isnumeric(LUT)
    if min(size(LUT)) == 1
        redlut      = LUT;
        greenlut    = LUT2;
        bluelut     = LUT3;
    elseif min(size(LUT)) == 2
        redlut      = LUT(:,2);
        greenlut    = LUT2(:,2);
        bluelut     = LUT3(:,2);
    else
        error('Please reconsider the format of your LUTs!');
    end
else    
    error('Your input Look-Up-Tables (LUT) are in the wrong format! Please check help of function!');
end;

% GAMMA CORRECTION = LOOK UP
% R
sz = size(R);
inds = ~isnan(R);
inds_nan = isnan(R);
R_corrected = redlut(R(inds) + 1);
R_corrected(inds_nan) = NaN;
R_corrected = reshape(R_corrected, sz(1), sz(2));
% G
sz = size(G);
inds = ~isnan(G);
inds_nan = isnan(G);
G_corrected = greenlut(G(inds) + 1);
G_corrected(inds_nan) = NaN;
G_corrected = reshape(G_corrected, sz(1), sz(2));
% B
sz = size(B);
inds = ~isnan(B);
inds_nan = isnan(B);
B_corrected = bluelut(B(inds) + 1);
B_corrected(inds_nan) = NaN;
B_corrected = reshape(B_corrected, sz(1), sz(2));

% WARNINGS IF VALUES OUT OF GAMMUT (necessary? 2009feb)
inds = find(R_corrected > 255);
R_corrected(inds) = 255;
inds2 = find(R_corrected < 0);
R_corrected(inds2) = 0;
if ~isempty(inds)| ~isempty(inds2)
    disp('overflows truncated to the range [0 255]')
end
inds = find(G_corrected > 255);
G_corrected(inds) = 255;
inds2 = find(G_corrected < 0);
G_corrected(inds2) = 0;
if ~isempty(inds)| ~isempty(inds2)
    disp('overflows truncated to the range [0 255]')
end
inds = find(B_corrected > 255);
B_corrected(inds) = 255;
inds2 = find(B_corrected < 0);
B_corrected(inds2) = 0;
if ~isempty(inds)| ~isempty(inds2)
    disp('overflows truncated to the range [0 255]')
end

%% gamut_checker
function [rgbl2, inds0, inds255] = gamutchecker(rgbl1, mon_oog, rgb_max, dim, wrng_detail)
%2014oct09 [cw]

if nargin < 5
    wrng_detail = 1;
    if nargin < 4
        dim = 2;
        if nargin < 3
            rgb_max = 255;
            if nargin < 2
                mon_oog = [0 255; 0 255; 0 255];
            end
        end
    end
end

n = size(rgbl1,1)*size(rgbl1,2);
rgbl2 = rgbl1;
if dim == 2
    for k = 1:size(rgbl2,2)
        ind1 = rgbl1(:,k) < mon_oog(k,1);
        inds0(:,k) = ind1;
        ind2 = rgbl1(:,k) > mon_oog(k,2);
        inds255(:,k) = ind2;
    end
    rgbl2(inds0) = 0;
    rgbl2(inds255) = rgb_max;
elseif dim == 3
    for k = 1:size(rgbl2,3)
        ind1 = rgbl1(:,:,k) < mon_oog(k,1);
        inds0(:,:,k) = ind1;
        ind2 = rgbl1(:,:,k) > mon_oog(k,2);
        inds255(:,:,k) = ind2;     
    end    
end
rgbl2(inds0) = 0;
rgbl2(inds255) = rgb_max;

if dim == 2
    indr = zeros(size(rgbl1,1),1);
    indg = zeros(size(rgbl1,1),1);
    indb = zeros(size(rgbl1,1),1);
    indr2 = zeros(size(rgbl1,1),1);
    indg2 = zeros(size(rgbl1,1),1);
    indb2 = zeros(size(rgbl1,1),1);
elseif dim == 3
    indr = zeros(size(rgbl1,1),size(rgbl1,2),1);
    indg = zeros(size(rgbl1,1),size(rgbl1,2),1);
    indb = zeros(size(rgbl1,1),size(rgbl1,2),1);
    indr2 = zeros(size(rgbl1,1),size(rgbl1,2),1);
    indg2 = zeros(size(rgbl1,1),size(rgbl1,2),1);
    indb2 = zeros(size(rgbl1,1),size(rgbl1,2),1);
end

if sum(inds0(:))>0 | sum(inds255(:))>0
    if dim == 2
        indr = rgbl1(:,1) < mon_oog(1,1);
        indg = rgbl1(:,2) < mon_oog(2,1);
        indb = rgbl1(:,3) < mon_oog(3,1);
        indr2 = rgbl1(:,1) > mon_oog(1,2);
        indg2 = rgbl1(:,2) > mon_oog(2,2);
        indb2 = rgbl1(:,3) > mon_oog(3,2);
        
    elseif dim == 3
        indr = rgbl1(:,:,1) < mon_oog(1,1);
        indg = rgbl1(:,:,2) < mon_oog(2,1);
        indb = rgbl1(:,:,3) < mon_oog(3,1);
        indr2 = rgbl1(:,:,1) > mon_oog(1,2);
        indg2 = rgbl1(:,:,2) > mon_oog(2,2);
        indb2 = rgbl1(:,:,3) > mon_oog(3,2);
    end
    if wrng_detail
        if sum(indr(:)) > 0
            fprintf('WARNING: R < 0 for %d.\n', sum(indr(:)));
        end
        if sum(indr2(:)) > 0
            fprintf('WARNING: R > %d for %d.\n', rgb_max, sum(indr2(:)));
        end
        if sum(indg(:)) > 0
            fprintf('WARNING: G < 0 for %d.\n', sum(indg(:)));
        end
        if sum(indg2(:)) > 0
            fprintf('WARNING: G > %d for %d.\n', rgb_max, sum(indg2(:)));
        end
        if sum(indb(:)) > 0
            fprintf('WARNING: B < 0 for %d.\n', sum(indb(:)));
        end
        if sum(indb2(:)) > 0
            fprintf('WARNING: B > %d for %d.\n', rgb_max, sum(indb2(:)));
        end
    end
    k0 = sum(any(inds0(:),dim));
    k255 = sum(any(inds255(:), dim));
    inds = inds0 | inds255;
    k = sum(any(inds(:),dim));
    if wrng_detail
        fprintf('WARNING: Overall %d <0 (%.1f%%) + %d >%d (%.1f%%) have been reset to gamut boundaries (%d of %d ~%.1f%%).\n', k0, (k0/n)*100, k255, rgb_max, (k255/n)*100, k, n, (k/n)*100);
    end
else
    if wrng_detail
        fprintf('OK: All rgb within gamut\n');
    end
end

%% hue_aligner
function [one_line, pc, explained, M] = hue_aligner(Dvec)
% 2018.11.19 * [cw]

[pc0,score,~,~,explained,mu] = pca(Dvec);
pc = pc0(:,1);
M = mu;

score(:,2:end) = 0;
one_line0 = score*inv(pc0);

for k = 1:size(one_line0,2)
    one_line(:,k)= one_line0(:,k)+M(k);
end

%% Luv2RGB
function RGBg = Luv2RGB(Luv, wp_XYZ, mon_xyY, mon_ldt, dim)
% 2019.10.23 * [cw]
if nargin < 5
    dim = 3;
end
[~, XYZ] = Luv2xyY(Luv, wp_XYZ, 'XYZ', dim);
if dim == 2
    [RGBl(:,1),RGBl(:,2),RGBl(:,3)] = XYZ2rgb(XYZ(:,1), XYZ(:,2), XYZ(:,3), mon_xyY);
else
    [RGBl(:,:,1),RGBl(:,:,2),RGBl(:,:,3)] = XYZ2rgb(XYZ(:,:,1), XYZ(:,:,2), XYZ(:,:,3), mon_xyY);
end
[RGBg] = gammacorrector(RGBl, mon_ldt, dim);

%% RGB2Luv
function Luv = RGB2Luv(RGBg, wp_XYZ, mon_xyY, mon_ldt)
% 2019.10.23 * [cw]
dim = 3;
[RGBl] = gammainvcorrector(RGBg, mon_ldt, dim, 255);
XYZ = rgb2XYZ(RGBl, mon_xyY, 3);
Luv = XYZ2Luv(XYZ, wp_XYZ, dim);

%% rgb2XYZ
function XYZ = rgb2XYZ(rgb, xyYmon, dim)
% 2018.03.29 black luminance correction added [cw]

xyYmon = xyYmon(1:3,1:3);
XYZmon = xyY2XYZ(xyYmon);

if dim == 2
    XYZ= rgb * XYZmon;
elseif dim == 3
    for dm = 1:3
        for nr = 1:3
            XYZ(:,:,nr,dm) = rgb(:,:,dm) * XYZmon(dm,nr);
        end
    end
    XYZ = sum(XYZ,4);
end

%% Luv2xyY
function [xyY, XYZ] = Luv2xyY(Luv, xyYn_or_XYZn, wp_mode, dim)
% 2012nov16 [cw]

if nargin < 4
    dim = 2;
    if nargin < 3
        % wp_mode
        wp_mode = 'xyY';
    end
end

% image mode (or not):
if dim == 2
        image_mode = 0;
        L = Luv(:,1); u = Luv(:,2); v = Luv(:,3);
elseif dim == 3
        image_mode = 1;
        L = Luv(:,:,1); u = Luv(:,:,2); v = Luv(:,:,3);
end

% WP
switch wp_mode
    case 'xyY'
        xn = xyYn_or_XYZn(:,1);
        yn = xyYn_or_XYZn(:,2);
        Yn = xyYn_or_XYZn(:,3);
        uprimen = (4 * xn)./(-2*xn + 12*yn + 3);
        vprimen = (9 * yn)./(-2*xn + 12*yn + 3);
    case 'XYZ'
        Xn = xyYn_or_XYZn(:,1);
        Yn = xyYn_or_XYZn(:,2);
        Zn = xyYn_or_XYZn(:,3);
        uprimen = (4 * Xn)./(Xn + 15 * Yn + 3 * Zn);
        vprimen = (9 * Yn)./(Xn + 15 * Yn + 3 * Zn);
    otherwise
        error('INPUT: wp_mode does not exist');
end


% uv2uv_
u_ = u./(13*L) + uprimen;
v_ = v./(13*L) + vprimen;


% uv_2xyY
x = (9 * u_)./(6*u_ - 16*v_ + 12);
y = (4 * v_)./(6*u_ - 16*v_ + 12);

Y = Yn * ((L+16)/116).^3;
Y2 = Yn * L * (3/29).^3;
inds = L<=8;
Y(inds) = Y2(inds);

if image_mode
    xyY = cat(3, x, y, Y);
else
    xyY = [x y Y];
end

% uv_2XYZ
if nargout > 1
    X = Y .* ((9*u_) ./ (4*v_));
    Z = Y .* (12 - 3*u_ - 20*v_)./(4*v_);
    
    % To correct for formula error with black:
    inds = L == 0;
    X(inds) = 0; Z(inds) = 0;

    if image_mode        
        XYZ = cat(3, X, Y, Z);
    else
        XYZ = [X Y Z];
    end
end

%% mask_2Dmapper
function Lxy2D = mask_2Dmapper(Lxy3D, mask)
%2018.12.14 * [cw]

L0 = Lxy3D(:,:,1);
x0 = Lxy3D(:,:,2);
y0 = Lxy3D(:,:,3);

L = L0(mask);
x = x0(mask);
y = y0(mask);

Lxy2D = [L,x,y]; 

%% mask_3Dmapper
function Lxy3D = mask_3Dmapper(Lxy, mask, bg)
%2018.11.20 * [cw]

if nargin < 3
    bg = [0 0 0];
end
sz = size(mask);
einzer = ones(sz(1),sz(2));

L2 = einzer*bg(1);
L2(mask) = Lxy(:,1);
x2 = einzer*bg(2);
x2(mask) = Lxy(:,2);
y2 = einzer*bg(3);
y2(mask) = Lxy(:,3);

Lxy3D = cat(3,L2,x2,y2);

%% srgb
function [mon_xyY, mon_ldt] = srgb
mon_ldt = [...
    0,0.077399,0.154799,0.232198,0.309598,0.386997,0.464396,0.541796,0.619195,0.696594,0.773994,0.853367,0.937509,1.026303,1.119818,1.218123,1.321287,1.429375,1.542452,1.660583,1.78383,1.912253,2.045914,2.184872,2.329185,2.47891,2.634105,2.794824,2.961123,3.133055,3.310673,3.494031,3.68318,3.878171,4.079055,4.285881,4.498698,4.717556,4.942502,5.173584,5.410848,5.654341,5.904108,6.160196,6.422649,6.691512,6.966827,7.24864,7.536993,7.831928,8.133488,8.441715,8.756651,9.078335,9.40681,9.742115,10.08429,10.4333750000000,10.78941,11.1524320000000,11.5224820000000,11.8995970000000,12.2838150000000,12.6751740000000,13.0737120000000,13.4794650000000,13.89247,14.3127650000000,14.7403850000000,15.1753660000000,15.6177440000000,16.0675550000000,16.5248330000000,16.9896140000000,17.4619330000000,17.9418240000000,18.4293220000000,18.92446,19.4272720000000,19.9377930000000,20.4560540000000,20.98209,21.5159340000000,22.0576180000000,22.6071750000000,23.1646360000000,23.7300360000000,24.3034040000000,24.8847740000000,25.4741760000000,26.0716420000000,26.6772030000000,27.2908910000000,27.9127360000000,28.5427690000000,29.18102,29.82752,30.4822990000000,31.1453870000000,31.8168130000000,32.4966090000000,33.1848020000000,33.8814220000000,34.5864990000000,35.3000620000000,36.0221390000000,36.75276,37.4919530000000,38.2397460000000,38.9961690000000,39.7612480000000,40.5350130000000,41.3174910000000,42.10871,42.9086970000000,43.7174810000000,44.5350880000000,45.3615460000000,46.1968820000000,47.0411240000000,47.8942970000000,48.7564290000000,49.6275470000000,50.5076760000000,51.3968450000000,52.2950780000000,53.2024020000000,54.1188430000000,55.0444280000000,55.9791810000000,56.9231290000000,57.8762980000000,58.8387120000000,59.8103980000000,60.7913810000000,61.7816860000000,62.7813380000000,63.7903630000000,64.8087840000000,65.8366270000000,66.8739180000000,67.9206790000000,68.9769370000000,70.0427150000000,71.1180370000000,72.2029290000000,73.2974140000000,74.4015160000000,75.5152590000000,76.6386680000000,77.7717650000000,78.9145750000000,80.0671220000000,81.2294280000000,82.4015180000000,83.5834150000000,84.7751420000000,85.9767220000000,87.1881780000000,88.4095340000000,89.6408130000000,90.8820370000000,92.1332290000000,93.3944120000000,94.6656090000000,95.9468410000000,97.2381330000000,98.5395060000000,99.8509820000000,101.172584000000,102.504334000000,103.846254000000,105.198366000000,106.560693000000,107.933256000000,109.316077000000,110.709177000000,112.112579000000,113.526305000000,114.950375000000,116.384811000000,117.829635000000,119.284868000000,120.750532000000,122.226647000000,123.713235000000,125.210317000000,126.717914000000,128.236047000000,129.764737000000,131.304005000000,132.853871000000,134.414357000000,135.985483000000,137.567270000000,139.159738000000,140.762907000000,142.376799000000,144.001434000000,145.636832000000,147.283012000000,148.939997000000,150.607804000000,152.286456000000,153.975971000000,155.676371000000,157.387673000000,159.1099,160.843070000000,162.587203000000,164.342319000000,166.108438000000,167.885578000000,169.673761000000,171.473005000000,173.283330000000,175.104755000000,176.937299000000,178.780982000000,180.635824000000,182.501843000000,184.379058000000,186.267489000000,188.167154000000,190.078073000000,192.000265000000,193.933749000000,195.878543000000,197.834666000000,199.802137000000,201.780975000000,203.771198000000,205.772826000000,207.785876000000,209.810367000000,211.846319000000,213.893748000000,215.952674000000,218.023115000000,220.105089000000,222.198615000000,224.303711000000,226.420395000000,228.548685000000,230.688599000000,232.840156000000,235.003373000000,237.178269000000,239.364861000000,241.563167000000,243.773205000000,245.994993000000,248.228549000000,250.473890000000,252.731035000000,255;...
    0,0.077399,0.154799,0.232198,0.309598,0.386997,0.464396,0.541796,0.619195,0.696594,0.773994,0.853367,0.937509,1.026303,1.119818,1.218123,1.321287,1.429375,1.542452,1.660583,1.78383,1.912253,2.045914,2.184872,2.329185,2.47891,2.634105,2.794824,2.961123,3.133055,3.310673,3.494031,3.68318,3.878171,4.079055,4.285881,4.498698,4.717556,4.942502,5.173584,5.410848,5.654341,5.904108,6.160196,6.422649,6.691512,6.966827,7.24864,7.536993,7.831928,8.133488,8.441715,8.756651,9.078335,9.40681,9.742115,10.08429,10.4333750000000,10.78941,11.1524320000000,11.5224820000000,11.8995970000000,12.2838150000000,12.6751740000000,13.0737120000000,13.4794650000000,13.89247,14.3127650000000,14.7403850000000,15.1753660000000,15.6177440000000,16.0675550000000,16.5248330000000,16.9896140000000,17.4619330000000,17.9418240000000,18.4293220000000,18.92446,19.4272720000000,19.9377930000000,20.4560540000000,20.98209,21.5159340000000,22.0576180000000,22.6071750000000,23.1646360000000,23.7300360000000,24.3034040000000,24.8847740000000,25.4741760000000,26.0716420000000,26.6772030000000,27.2908910000000,27.9127360000000,28.5427690000000,29.18102,29.82752,30.4822990000000,31.1453870000000,31.8168130000000,32.4966090000000,33.1848020000000,33.8814220000000,34.5864990000000,35.3000620000000,36.0221390000000,36.75276,37.4919530000000,38.2397460000000,38.9961690000000,39.7612480000000,40.5350130000000,41.3174910000000,42.10871,42.9086970000000,43.7174810000000,44.5350880000000,45.3615460000000,46.1968820000000,47.0411240000000,47.8942970000000,48.7564290000000,49.6275470000000,50.5076760000000,51.3968450000000,52.2950780000000,53.2024020000000,54.1188430000000,55.0444280000000,55.9791810000000,56.9231290000000,57.8762980000000,58.8387120000000,59.8103980000000,60.7913810000000,61.7816860000000,62.7813380000000,63.7903630000000,64.8087840000000,65.8366270000000,66.8739180000000,67.9206790000000,68.9769370000000,70.0427150000000,71.1180370000000,72.2029290000000,73.2974140000000,74.4015160000000,75.5152590000000,76.6386680000000,77.7717650000000,78.9145750000000,80.0671220000000,81.2294280000000,82.4015180000000,83.5834150000000,84.7751420000000,85.9767220000000,87.1881780000000,88.4095340000000,89.6408130000000,90.8820370000000,92.1332290000000,93.3944120000000,94.6656090000000,95.9468410000000,97.2381330000000,98.5395060000000,99.8509820000000,101.172584000000,102.504334000000,103.846254000000,105.198366000000,106.560693000000,107.933256000000,109.316077000000,110.709177000000,112.112579000000,113.526305000000,114.950375000000,116.384811000000,117.829635000000,119.284868000000,120.750532000000,122.226647000000,123.713235000000,125.210317000000,126.717914000000,128.236047000000,129.764737000000,131.304005000000,132.853871000000,134.414357000000,135.985483000000,137.567270000000,139.159738000000,140.762907000000,142.376799000000,144.001434000000,145.636832000000,147.283012000000,148.939997000000,150.607804000000,152.286456000000,153.975971000000,155.676371000000,157.387673000000,159.1099,160.843070000000,162.587203000000,164.342319000000,166.108438000000,167.885578000000,169.673761000000,171.473005000000,173.283330000000,175.104755000000,176.937299000000,178.780982000000,180.635824000000,182.501843000000,184.379058000000,186.267489000000,188.167154000000,190.078073000000,192.000265000000,193.933749000000,195.878543000000,197.834666000000,199.802137000000,201.780975000000,203.771198000000,205.772826000000,207.785876000000,209.810367000000,211.846319000000,213.893748000000,215.952674000000,218.023115000000,220.105089000000,222.198615000000,224.303711000000,226.420395000000,228.548685000000,230.688599000000,232.840156000000,235.003373000000,237.178269000000,239.364861000000,241.563167000000,243.773205000000,245.994993000000,248.228549000000,250.473890000000,252.731035000000,255;...
    0,0.077399,0.154799,0.232198,0.309598,0.386997,0.464396,0.541796,0.619195,0.696594,0.773994,0.853367,0.937509,1.026303,1.119818,1.218123,1.321287,1.429375,1.542452,1.660583,1.78383,1.912253,2.045914,2.184872,2.329185,2.47891,2.634105,2.794824,2.961123,3.133055,3.310673,3.494031,3.68318,3.878171,4.079055,4.285881,4.498698,4.717556,4.942502,5.173584,5.410848,5.654341,5.904108,6.160196,6.422649,6.691512,6.966827,7.24864,7.536993,7.831928,8.133488,8.441715,8.756651,9.078335,9.40681,9.742115,10.08429,10.4333750000000,10.78941,11.1524320000000,11.5224820000000,11.8995970000000,12.2838150000000,12.6751740000000,13.0737120000000,13.4794650000000,13.89247,14.3127650000000,14.7403850000000,15.1753660000000,15.6177440000000,16.0675550000000,16.5248330000000,16.9896140000000,17.4619330000000,17.9418240000000,18.4293220000000,18.92446,19.4272720000000,19.9377930000000,20.4560540000000,20.98209,21.5159340000000,22.0576180000000,22.6071750000000,23.1646360000000,23.7300360000000,24.3034040000000,24.8847740000000,25.4741760000000,26.0716420000000,26.6772030000000,27.2908910000000,27.9127360000000,28.5427690000000,29.18102,29.82752,30.4822990000000,31.1453870000000,31.8168130000000,32.4966090000000,33.1848020000000,33.8814220000000,34.5864990000000,35.3000620000000,36.0221390000000,36.75276,37.4919530000000,38.2397460000000,38.9961690000000,39.7612480000000,40.5350130000000,41.3174910000000,42.10871,42.9086970000000,43.7174810000000,44.5350880000000,45.3615460000000,46.1968820000000,47.0411240000000,47.8942970000000,48.7564290000000,49.6275470000000,50.5076760000000,51.3968450000000,52.2950780000000,53.2024020000000,54.1188430000000,55.0444280000000,55.9791810000000,56.9231290000000,57.8762980000000,58.8387120000000,59.8103980000000,60.7913810000000,61.7816860000000,62.7813380000000,63.7903630000000,64.8087840000000,65.8366270000000,66.8739180000000,67.9206790000000,68.9769370000000,70.0427150000000,71.1180370000000,72.2029290000000,73.2974140000000,74.4015160000000,75.5152590000000,76.6386680000000,77.7717650000000,78.9145750000000,80.0671220000000,81.2294280000000,82.4015180000000,83.5834150000000,84.7751420000000,85.9767220000000,87.1881780000000,88.4095340000000,89.6408130000000,90.8820370000000,92.1332290000000,93.3944120000000,94.6656090000000,95.9468410000000,97.2381330000000,98.5395060000000,99.8509820000000,101.172584000000,102.504334000000,103.846254000000,105.198366000000,106.560693000000,107.933256000000,109.316077000000,110.709177000000,112.112579000000,113.526305000000,114.950375000000,116.384811000000,117.829635000000,119.284868000000,120.750532000000,122.226647000000,123.713235000000,125.210317000000,126.717914000000,128.236047000000,129.764737000000,131.304005000000,132.853871000000,134.414357000000,135.985483000000,137.567270000000,139.159738000000,140.762907000000,142.376799000000,144.001434000000,145.636832000000,147.283012000000,148.939997000000,150.607804000000,152.286456000000,153.975971000000,155.676371000000,157.387673000000,159.1099,160.843070000000,162.587203000000,164.342319000000,166.108438000000,167.885578000000,169.673761000000,171.473005000000,173.283330000000,175.104755000000,176.937299000000,178.780982000000,180.635824000000,182.501843000000,184.379058000000,186.267489000000,188.167154000000,190.078073000000,192.000265000000,193.933749000000,195.878543000000,197.834666000000,199.802137000000,201.780975000000,203.771198000000,205.772826000000,207.785876000000,209.810367000000,211.846319000000,213.893748000000,215.952674000000,218.023115000000,220.105089000000,222.198615000000,224.303711000000,226.420395000000,228.548685000000,230.688599000000,232.840156000000,235.003373000000,237.178269000000,239.364861000000,241.563167000000,243.773205000000,245.994993000000,248.228549000000,250.473890000000,252.731035000000,255]';

mon_xyY = [...
    0.64,0.33,17;...
    0.30,0.60,57.2;...
    0.15,0.060,5.8;...
    0.3127,0.329,80];

%% xyY2Luv
function Luv = xyY2Luv(xyY, xyY_bg, dim)
%   2012aug06 [cw]

if nargin < 3
    dim = 2;
end

XYZ = xyY2XYZ(xyY, dim);
XYZ_bg = xyY2XYZ(xyY_bg, 2);

Luv = XYZ2Luv(XYZ, XYZ_bg, dim);

%% xyY2XYZ
function XYZ = xyY2XYZ(xyY, dim)
% 2018.07.23 [cw]

if nargin < 2
    dim = 2;
end

if dim == 2
    x = xyY(:,1); y = xyY(:,2); Y = xyY(:,3);
elseif dim == 3
    x = xyY(:,:,1); y = xyY(:,:,2); Y = xyY(:,:,3); 
end

X = (Y./y) .* x;
Y = Y;
Z = (Y./y) .* (1-y-x);

if dim == 2
    XYZ = [X Y Z];
    inds = XYZ(:,2) == 0;
    XYZ(inds,:) = 0;
elseif dim == 3
    inds = Y == 0;
    X(inds) = 0; Z(inds) = 0;
    XYZ = cat(3, X, Y, Z);    
end

%% XYZ2rgb
function [R_or_RGB, G, B] = XYZ2rgb(X_or_XYZ, Y_or_xyYmon, Z, xyYmon, rgbmax)
%   2017.12.20 [cw].

if nargin < 5
    rgbmax = 255;
    if nargin < 2
        xyYmon = srgb;
        if nargin < 1;
            show_input_structure; % subfunction, s. below.
            return;
        end
    end
end

if nargin == 2 % Vector mode
    XYZ = X_or_XYZ;
    xyYmon = Y_or_xyYmon;
    %the coordinates have to be translated to tristimulus values
    [XYZmon] = xyY2XYZ(xyYmon);
%    rgb = ((inv(XYZmon')*XYZ')*rgbmax)';
    rgb = ((XYZmon'\XYZ')*rgbmax)';
    %warning in case values are outside the gamut
     if  ~isempty(find(rgb < 0)) || ~isempty(find(rgb > rgbmax))  
        warning('xyY values outside display gamut')
     end
     R_or_RGB = rgb; % The output R_or_RGB contains the RGB-values.
elseif nargin >= 4 % Matrices mode
    X = X_or_XYZ; % Input X_or_XYZ contains the matrix with the X values. 
    Y = Y_or_xyYmon; % Input Y_or_xyYmon contains the matrix with the Y values.
    %the coordinates have to be translated to tristimulus values
    [XYZmon] = xyY2XYZ(xyYmon); 
    
    s_r = size(X,1);
    s_c = size(X,2);

    XYZ = [reshape(X, s_r*s_c, 1), reshape(Y, s_r*s_c, 1), reshape(Z, s_r*s_c, 1)];
    inverse = inv(XYZmon(1:3,:)');
    RGB = ((inverse*XYZ')*rgbmax)';

    R_v = RGB(:,1);
    R = reshape(R_v, s_r,s_c);
    G_v = RGB(:,2);
    G = reshape(G_v, s_r,s_c);
    B_v = RGB(:,3);
    B = reshape(B_v, s_r,s_c);
    R_or_RGB = R; % The output R_or_RGB contains the matrix with R-values.

    %warning in case values are outside the gamut
    if  any((any(R<0)|any(R>rgbmax) | any(G<0)|any(G>rgbmax) | any(B<0)| any(B>rgbmax)))
%        warning('xyY values outside display gamut')
    end
else
    error('Check number of input & output!')
end

%% XYZ2Luv
function LUV = XYZ2Luv(XYZ, XYZ_bg, dim)
% 2016.07.01 [cw]

if nargin < 3
    dim = 2;
end

% Whitepoint:
Xn = XYZ_bg(1);
Yn = XYZ_bg(2);
Zn = XYZ_bg(3);


if dim == 2
    X = XYZ(:,1);
    Y = XYZ(:,2);
    Z = XYZ(:,3);
elseif dim == 3
    X = XYZ(:,:,1);
    Y = XYZ(:,:,2);
    Z = XYZ(:,:,3);
end


uprime = (4 * X)./(X + 15 * Y + 3 * Z);
vprime = (9 * Y)./(X + 15 * Y + 3 * Z);
uprimen = (4 * Xn)./(Xn + 15 * Yn + 3 * Zn);
vprimen = (9 * Yn)./(Xn + 15 * Yn + 3 * Zn); 

L = 116 * (Y/Yn).^(1/3) - 16;
L2 = ((29/3)^3) * (Y/Yn);
inds = Y/Yn > (6/29)^3;
L(~inds) = L2(~inds);

u = (13*L) .* (uprime - uprimen);
v = (13*L) .* (vprime - vprimen);

% CORRECT FOR BLACK -------------------------------------------------------
inds = L == 0;
u(inds) = 0;
v(inds) = 0;

if dim == 2
    LUV = [L u v];
elseif dim == 3
    LUV = cat(3, L, u, v);    
end

%% XZY2xyY
function [xyY, xyz] = XYZ2xyY(XYZ, dim)
% 2014feb10 added 2. output xyz

if nargin < 2
    dim = 2;
end


if dim == 2
    X = XYZ(:, 1);Y = XYZ(:,2); Z = XYZ(:, 3);
elseif dim == 3
    X = XYZ(:,:,1); Y = XYZ(:,:,2); Z = XYZ(:,:,3);
end

x = X./(X+Y+Z);
y = Y./(X+Y+Z);
z = Z./(X+Y+Z);

if dim == 2
    xyY = [x y Y];
    xyz = [x y z];
elseif dim == 3
    xyY = cat(3, x, y, Y);
    xyz = cat(3, x, y, z);
end
