addpath(genpath(fileparts(mfilename('fullpath'))));
dataPath = fullfile('..','data','measurements.csv');
if ~isfile(dataPath)
    error('Data file not found: %s', dataPath);
end
T = step1_load_measurements(dataPath);
ocidPath = fullfile('..','data','opencellid_towers.csv');
demPath = fullfile('..','data','dem.tif');
T = step2_feature_engineer(T, ocidPath, demPath);
resMeters = 100;
[Finterp, Ggrid, gx, gy, proj] = step3_baseline_interpolation(T, resMeters);
doOptimize = true; 
modelEnsemble = step3_ml_ensemble(T, doOptimize);
K = 5;
stats = step5_spatial_cv(T, K, @(Ttrain) step3_ml_ensemble(Ttrain, false));
disp('Spatial CV results (per fold):');
disp(stats);
modelArgs = struct( ...
  'latTx',    mean(T.latitude), ...
  'lonTx',    mean(T.longitude), ...
  'txPowerdBm', 30, ...
  'txHeightM',  30, ...
  'freqHz',     1.8e9, ...
  'rxHeightM',  1.5, ...
  'modelName', 'longley-rice');
results = step4_generate_compare_maps(T, Finterp, proj, modelEnsemble, modelArgs);
disp(results);
out_interp = fullfile('..','data','coverage_interpolated.tif');
step6_export_geotiff(out_interp, gx, gy, Ggrid, proj);
try
    [latGrid, lonGrid] = projinv(proj, gx, gy);
catch
    [latGrid, lonGrid] = projinv(proj, gx', gy'); latGrid = latGrid'; lonGrid = lonGrid';
end
npts = numel(latGrid);
Xgrid = [latGrid(:), lonGrid(:), zeros(npts,1), zeros(npts,1), zeros(npts,1)];
coords_meas = [T.latitude, T.longitude];
for i=1:npts
    [~, idx] = min(sum((coords_meas - Xgrid(i,1:2)).^2,2));
    Xgrid(i,3) = T.altitude(idx);
    Xgrid(i,4) = T.dist_to_tower_m(idx);
    Xgrid(i,5) = T.dem_elev_m(idx);
end
Gensemble = predict(modelEnsemble, Xgrid);
Gensemble = reshape(Gensemble, size(gx));
out_ens = fullfile('..','data','coverage_ensemble.tif');
step6_export_geotiff(out_ens, gx, gy, Gensemble, proj);
webOutFolder = fullfile('..','data','webmap');
if ~exist(webOutFolder,'dir'), mkdir(webOutFolder); 
end
step7_export_webmap(out_interp, webOutFolder, 'coverage_interpolated_map.html');
step7_export_webmap(out_ens, webOutFolder, 'coverage_ensemble_map.html');
disp('All done. Open the HTML files in the data/webmap folder to view results.');
function T = step1_load_measurements(csvPath)
T = readtable(csvPath);
T.Properties.VariableNames = lower(T.Properties.VariableNames);
required = {'latitude','longitude','signal'};
if ~all(ismember(required, T.Properties.VariableNames))
    error('CSV must contain latitude, longitude, signal columns.');
end
if ~ismember('altitude', T.Properties.VariableNames)
    T.altitude = zeros(height(T),1);
end
T = rmmissing(T);
T.signal = min(max(T.signal, -140), -30);
end
function T = step2_feature_engineer(T, ocidPath, demPath)
T.dist_to_tower_m = nan(height(T),1);
T.dem_elev_m = nan(height(T),1);
if nargin>=2 && isfile(ocidPath)
    try
        towers = readtable(ocidPath);
        towers.Properties.VariableNames = lower(towers.Properties.VariableNames);
        if ~all(ismember({'latitude','longitude'}, towers.Properties.VariableNames))
            warning('OpenCellID CSV must contain latitude, longitude columns. Skipping tower features.');
        else
            for i=1:height(T)
                lat1 = T.latitude(i); lon1 = T.longitude(i);
                d = haversine_distance(lat1, lon1, towers.latitude, towers.longitude);
                T.dist_to_tower_m(i) = min(d);
            end
        end
    catch ME
       warning("Error reading OpenCellID file: " + ME.message + ". Skipping tower features.");
    end
end
if nargin>=3 && isfile(demPath)
    try
        [Z, R] = readgeoraster(demPath,'OutputType','double');
        Z = double(Z);
        for i=1:height(T)
            lat = T.latitude(i); lon = T.longitude(i);
            try
                [row, col] = geographicToDiscrete(R, lat, lon);
                if row>=1 && row<=size(Z,1) && col>=1 && col<=size(Z,2)
                    T.dem_elev_m(i) = Z(row,col);
                else
                    T.dem_elev_m(i) = NaN;
                end
            catch
                T.dem_elev_m(i) = NaN;
            end
        end
    catch ME
        warning("Error reading DEM GeoTIFF: " + ME.message + ". Skipping DEM features.");
    end
end
T.dist_to_tower_m(isnan(T.dist_to_tower_m)) = median(T.dist_to_tower_m(~isnan(T.dist_to_tower_m)), 'omitnan');
if isempty(T.dist_to_tower_m) || all(isnan(T.dist_to_tower_m))
    T.dist_to_tower_m = repmat(mean(T.dist_to_tower_m,'omitnan'), height(T),1);
end
T.dem_elev_m(isnan(T.dem_elev_m)) = T.altitude(isnan(T.dem_elev_m));

end
function d = haversine_distance(lat1, lon1, lat2, lon2)
R = 6371000; 
phi1 = deg2rad(lat1);
phi2 = deg2rad(lat2);
dphi = deg2rad(lat2 - lat1);
dlambda = deg2rad(lon2 - lon1);
a = sin(dphi/2).^2 + cos(phi1).*cos(phi2).*sin(dlambda/2).^2;
c = 2.*atan2(sqrt(a), sqrt(1-a));
d = R.*c;
end
function [F, G, gx, gy, proj] = step3_baseline_interpolation(T, resMeters)
[wx, wy, proj] = utils_geo.latlon2local(T.latitude, T.longitude);
pad = 500;
xv = (min(wx)-pad):resMeters:(max(wx)+pad);
yv = (min(wy)-pad):resMeters:(max(wy)+pad);
[gx, gy] = meshgrid(xv, yv);
F = scatteredInterpolant(wx, wy, T.signal, 'natural', 'nearest');
G = F(gx, gy); 
end
function M = step3_ml_ensemble(T, doOptimize)
if nargin<2, doOptimize = false; end
X = [T.latitude, T.longitude, T.altitude, T.dist_to_tower_m, T.dem_elev_m];
y = T.signal;
for j=1:size(X,2)
    col = X(:,j);
    if any(isnan(col))
        col(isnan(col)) = median(col(~isnan(col)));
        X(:,j) = col;
    end
end
cvp = cvpartition(size(X,1),'HoldOut',0.2);
Xtr = X(training(cvp),:); ytr = y(training(cvp));
Xte = X(test(cvp),:);     yte = y(test(cvp));

if doOptimize
    try
        fprintf('Starting hyperparameter optimization (reduced budget)...\n');
        hyopts = struct('AcquisitionFunctionName','expected-improvement-plus', ...
                        'MaxObjectiveEvaluations',8, 'ShowPlots',false, 'Verbose',0);
        M = fitrensemble(Xtr,ytr,'Method','LSBoost','OptimizeHyperparameters',{'NumLearningCycles','LearnRate','MinLeafSize'}, ...
            'HyperparameterOptimizationOptions', hyopts, 'Learners', templateTree('Reproducible',true));
    catch ME
        warning(['Optimization failed: ' char(ME.message) '. Falling back to sensible defaults.'])
        M = fitrensemble(Xtr,ytr,'Method','LSBoost','NumLearningCycles',100,'LearnRate',0.05, 'Learners', templateTree('Reproducible',true));
    end
else
    M = fitrensemble(Xtr,ytr,'Method','LSBoost','NumLearningCycles',100,'LearnRate',0.05, 'Learners', templateTree('Reproducible',true));
end
yp = predict(M, Xte);
utils_eval.print_regression_metrics(yte, yp);
end
function results = step4_generate_compare_maps(measT, Finterp, proj, modelRF, modelArgs)
[wx, wy] = projfwd(proj, measT.latitude, measT.longitude);
ss_interp = Finterp(wx, wy);
X = [measT.latitude, measT.longitude, measT.altitude, measT.dist_to_tower_m, measT.dem_elev_m];
for j=1:size(X,2)
    col = X(:,j);
    if any(isnan(col))
        col(isnan(col)) = median(col(~isnan(col)));
        X(:,j) = col;
    end
end
ss_rf = predict(modelRF, X);
try
    tx = txsite('Latitude',modelArgs.latTx,'Longitude',modelArgs.lonTx, ...
        'AntennaHeight',modelArgs.txHeightM,'TransmitterPower',modelArgs.txPowerdBm);
    rx = rxsite('Latitude', measT.latitude, 'Longitude', measT.longitude, ...
        'AntennaHeight', modelArgs.rxHeightM);
    pm = propagationModel(modelArgs.modelName);
    ss_theory = sigstrength(rx, tx, pm);
catch
    ss_theory = nan(height(measT),1);
    warning('Theoretical model not computed (txsite/propagation toolbox maybe missing).');
end
y = measT.signal;
R_interp = utils_eval.metrics(y, ss_interp);
R_rf = utils_eval.metrics(y, ss_rf);
if all(~isnan(ss_theory))
    R_theory = utils_eval.metrics(y, ss_theory);
else
    R_theory = struct('RMSE',nan,'MAE',nan,'R2',nan);
end

results = table( ...
  ['Interpolation';'RandomForest';'Theoretical'], ...
  [R_interp.RMSE; R_rf.RMSE; R_theory.RMSE], ...
  [R_interp.MAE;  R_rf.MAE;  R_theory.MAE], ...
  [R_interp.R2;   R_rf.R2;   R_theory.R2], ...
  'VariableNames', {'Model','RMSE','MAE','R2'});
utils_viz.plot_geoscatter(measT.latitude, measT.longitude, y, 'Measured (dBm)');
utils_viz.plot_geoscatter(measT.latitude, measT.longitude, ss_interp, 'Interpolated (dBm)');
utils_viz.plot_geoscatter(measT.latitude, measT.longitude, ss_rf, 'Random Forest (dBm)');
end
function stats = step5_spatial_cv(T, K, modelFcn)
if nargin<2, K = 5; end
if nargin<3, modelFcn = []; end
[wx, wy, ~] = utils_geo.latlon2local(T.latitude, T.longitude);
XY = [wx, wy];
rng(1);
[idx, ~] = kmeans(XY, K, 'Replicates',5);
rmse_interp = zeros(K,1);
mae_interp = zeros(K,1);
r2_interp = zeros(K,1);
rmse_model = zeros(K,1);
mae_model = zeros(K,1);
r2_model = zeros(K,1);
for k=1:K
    testIdx = (idx==k);
    trainIdx = ~testIdx;
    Ttrain = T(trainIdx,:);
    Ttest  = T(testIdx,:);
  [Ftrain, ~, ~, proj] = step3_baseline_interpolation(Ttrain, 100);
    try
        [wx_t, wy_t] = projfwd(proj, Ttest.latitude, Ttest.longitude);
    catch
        [wx_t, wy_t, ~] = utils_geo.latlon2local(Ttest.latitude, Ttest.longitude);
    end
    y_pred_interp = Ftrain(wx_t, wy_t);
    y_true = Ttest.signal;
    S = utils_eval.metrics(y_true, y_pred_interp);
    rmse_interp(k) = S.RMSE; mae_interp(k) = S.MAE; r2_interp(k) = S.R2;
    if ~isempty(modelFcn)
        M = modelFcn(Ttrain);
        Xtest = [Ttest.latitude, Ttest.longitude, Ttest.altitude, Ttest.dist_to_tower_m, Ttest.dem_elev_m];
        for j=1:size(Xtest,2)
            col = Xtest(:,j);
            if any(isnan(col)), col(isnan(col)) = median(col(~isnan(col))); Xtest(:,j)=col; 
            end
        end
        y_pred_model = predict(M, Xtest);
        S2 = utils_eval.metrics(y_true, y_pred_model);
        rmse_model(k) = S2.RMSE; mae_model(k) = S2.MAE; r2_model(k) = S2.R2;
    else
        rmse_model(k) = NaN; mae_model(k) = NaN; r2_model(k) = NaN;
    end
    fprintf('Fold %d/%d: interp RMSE=%.2f, model RMSE=%.2f\n', k, K, rmse_interp(k), rmse_model(k));
end

stats = struct();
stats.interp.RMSE = rmse_interp; stats.interp.MAE = mae_interp; stats.interp.R2 = r2_interp;
stats.model.RMSE = rmse_model; stats.model.MAE = mae_model; stats.model.R2 = r2_model;
end
function step6_export_geotiff(filename, gx, gy, G, proj)
try
    if ~isempty(proj)
        try
            [latGrid, lonGrid] = projinv(proj, gx, gy);
        catch
            try
                [latGrid, lonGrid] = projinv(proj, gx', gy');
                latGrid = latGrid'; 
                lonGrid = lonGrid';
            catch
                error('projinv failed');
            end
        end
    else
        error('No projection available for accurate georeferencing.');
    end

    latlim = [min(latGrid(:)) max(latGrid(:))];
    lonlim = [min(lonGrid(:)) max(lonGrid(:))];
    R = georefcells(latlim, lonlim, size(G));
    dataOut = single(G);
    geotiffwrite(filename, dataOut, R);
    fprintf('GeoTIFF written: %s\n', filename);

catch ME
    warning("Failed to write GeoTIFF: " + ME.message);
end
end
function step7_export_webmap(geotiffPath, outFolder, outHTMLName)
if nargin<3, outHTMLName = 'coverage_map.html'; end
if ~isfile(geotiffPath)
    error('GeoTIFF not found: %s', geotiffPath);
end
if ~exist(outFolder, 'dir'), mkdir(outFolder); 
end
[A, R] = readgeoraster(geotiffPath);
A = single(A);
A(isnan(A)) = min(A(:));
vmin = prctile(A(:), 2);
vmax = prctile(A(:), 98);
Aclipped = min(max(A, vmin), vmax);
Ascaled = uint8(255 * (Aclipped - vmin) ./ (vmax - vmin));
if ismatrix(Ascaled)
    cmap = jet(256);
    rgb = ind2rgb(double(Ascaled)+1, cmap);
    rgb = uint8(rgb * 255);
else
    rgb = Ascaled;
end
rgb = flipud(rgb);
pngName = fullfile(outFolder, 'coverage_overlay.png');
imwrite(rgb, pngName, 'PNG');
latlim = R.LatitudeLimits;
lonlim = R.LongitudeLimits;
bounds = [latlim(1), lonlim(1); latlim(2), lonlim(2)];
htmlPath = fullfile(outFolder, outHTMLName);
fid = fopen(htmlPath, 'w');
fprintf(fid, ['<!doctype html>\n<html>\n<head>\n<meta charset="utf-8"/>\n<title>Coverage Map</title>\n' ...
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n' ...
    '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />\n' ...
    '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>\n' ...
    '<style>html,body,#map{height:100%%;margin:0;padding:0}</style>\n</head>\n<body>\n' ...
    '<div id="map"></div>\n<script>var map=L.map("map").setView([%f,%f], 13);\n' ...
    'L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",{maxZoom:19}).addTo(map);\n' ...
    'var img = "%s";\nvar bounds = [[%f,%f],[%f,%f]];\nL.imageOverlay(img, bounds).addTo(map);\nmap.fitBounds(bounds);\n</script>\n</body>\n</html>'], ...
    mean(latlim), mean(lonlim), 'coverage_overlay.png', bounds(1,1), bounds(1,2), bounds(2,1), bounds(2,2));
fclose(fid);
fprintf('Web map created: %s\nOpen %s in a browser (from folder %s)\n', htmlPath, outHTMLName, outFolder);
end
 

