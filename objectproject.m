% objBehaviour_Project.m
clear; 
close all;
clc;
cfg.StopTime        = 30;        
cfg.SampleTime      = 0.1;        
cfg.SensorMaxRange  = 80;        
cfg.CameraProb      = 0.95;       
cfg.NumSequences    = 300;        
cfg.SequenceLen     = 40;        
cfg.FeatureWindow   = 5;          
rng(0);                          
disp('--- Building scenario template ---');
templateScenario = createTemplateScenario(cfg.StopTime, cfg.SampleTime);
cam = visionDetectionGenerator( ...
    'SensorIndex',1, ...
    'SensorLocation',[1.2 0], ...
    'Height',1.1, ...
    'MaxRange',cfg.SensorMaxRange, ...
    'HasOcclusion',true, ...
    'DetectionProbability',cfg.CameraProb);
rad = radarDetectionGenerator( ...
    'SensorIndex',2, ...
    'MountingLocation',[0.8 0 0.5], ...
    'FieldOfView',[20 5], ...
    'DetectionProbability',cfg.RadarProb, ...
    'MaxRange',cfg.SensorMaxRange, ...
    'HasRangeRate',true, ...
    'AzimuthResolution',4);
disp('--- Generating dataset ---');

seqFeatures = {};     
seqLabels   = [];      
frameFeat   = [];    
for s = 1:cfg.NumSequences
    seed = randi(10000);
    [sc, ainfo, ego] = spawnScene(templateScenario, 'urban_crossing', seed); 
    tStep = cfg.SampleTime;
    nFrames = cfg.SequenceLen;
    lastVel = containers.Map('KeyType','double','ValueType','any');
    frameBuffer = []; 
    for f = 1:nFrames
        advance(sc, tStep);
        for ai = 1:numel(ainfo.Actors)
            a   = ainfo.Actors(ai);
            pos = a.Position;   
            vel = a.Velocity;    
            spd = hypot(vel(1),vel(2));
            lat = pos(2);       
            key = double(ai);
            if isKey(lastVel, key)
                vprev = lastVel(key);
                acc = hypot(vel(1)-vprev(1), vel(2)-vprev(2)) / tStep;
            else
                acc = 0;
            end
            lastVel(key) = vel;
            label = double(ainfo.IsRisky(ai)); 

            frameBuffer = [frameBuffer; s f ai pos(1) pos(2) vel(1) vel(2) spd lat acc label]; %#ok<AGROW>
        end
    end
    for ai = 1:numel(ainfo.Actors)
        rows = frameBuffer(frameBuffer(:,3)==ai, :);
        seqMat = rows(:,[8 9 6 7 10]);
        seqFeatures{end+1} = seqMat; %#ok<SAGROW>
        seqLabels(end+1,1) = rows(1,end); %#ok<SAGROW>
    end
    W = cfg.FeatureWindow;
    for fi = 1:size(frameBuffer,1)
        sID = frameBuffer(fi,1); fr = frameBuffer(fi,2); ai = frameBuffer(fi,3);
        rows = frameBuffer(frameBuffer(:,1)==sID & frameBuffer(:,3)==ai & frameBuffer(:,2) <= fr, :);
        rows = rows(max(1,size(rows,1)-W+1):end,:); % last W
        meanSpeed = mean(rows(:,8));
        stdSpeed  = std(rows(:,8));
        latChange = rows(end,9) - rows(1,9);
        meanAcc   = mean(rows(:,10));
        vx = rows(end,6); vy = rows(end,7);
        label = rows(end,11);
        frameFeat = [frameFeat; meanSpeed stdSpeed latChange meanAcc vx vy label];%#ok<AGROW> 
    end
    delete(sc);
end

disp('--- Dataset generation complete ---');
X_frame = frameFeat(:,1:6);
y_frame = frameFeat(:,7);
X_seq = seqFeatures;
y_seq = seqLabels;
disp('--- Training classical classifier (SVM) ---');
cv = cvpartition(y_frame,'HoldOut',0.2);
Xtr = X_frame(training(cv),:); ytr = y_frame(training(cv));
Xte = X_frame(test(cv),:);     yte = y_frame(test(cv));
mu = mean(Xtr,1); sigma = std(Xtr,[],1);
XtrS = (Xtr - mu) ./ max(1e-6, sigma);
XteS = (Xte - mu) ./ max(1e-6, sigma);
svmModel = fitcsvm(XtrS, ytr, ...
    'KernelFunction','rbf', ...
    'KernelScale','auto', ...
    'Standardize',false, ...
    'ClassNames',[0 1]);
predSVM = predict(svmModel, XteS);
cm = confusionmat(yte, predSVM);
acc_svm = sum(diag(cm))/sum(cm(:));
fprintf('SVM Accuracy: %.3f  | Confusion [TN FP; FN TP] = [%d %d; %d %d]\n', ...
    acc_svm, cm(1,1),cm(1,2),cm(2,1),cm(2,2));
disp('--- Preparing & training LSTM ---');
numSeq   = numel(X_seq);
inputDim = size(X_seq{1},2); 
for i = 1:numSeq
    X_seq{i} = X_seq{i}';   
end
idx   = randperm(numSeq);
nTr   = round(0.8*numSeq);
XTrain = X_seq(idx(1:nTr));
YTrain = categorical(y_seq(idx(1:nTr)));
XTest  = X_seq(idx(nTr+1:end));
YTest  = categorical(y_seq(idx(nTr+1:end)));
layers = [ ...
    sequenceInputLayer(inputDim)
    bilstmLayer(64,'OutputMode','last')
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',12, ...
    'MiniBatchSize',64, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'Verbose',false);
lstmNet = trainNetwork(XTrain, YTrain, layers, options);
YPred = classify(lstmNet, XTest, 'SequenceLength','longest');
acc_lstm = mean(YPred == YTest);
fprintf('LSTM (sequence) Accuracy: %.3f\n', acc_lstm);
disp('--- Tracking + online classification demo ---');
filtInitFcn = @(detection) initConstVel2D(detection, cfg.SampleTime);
tracker = trackerGNN('FilterInitializationFcn', filtInitFcn, ...
                     'ConfirmationThreshold',[3 5], ...
                     'DeletionThreshold',8, ...
                     'AssignmentThreshold',30);
[sc_eval, ainfo_eval, ego_eval] = spawnScene(templateScenario, 'urban_crossing', 9999);
trackHist     = containers.Map('KeyType','double','ValueType','any');  
trackLabelsGT = containers.Map('KeyType','double','ValueType','double'); 
ev.true = []; ev.pred = [];
while sc_eval.SimulationTime < cfg.SampleTime * cfg.SequenceLen
    advance(sc_eval, cfg.SampleTime);
    tgtPoses  = actorPoses(sc_eval);
    egoPose   = targetPoses(ego_eval);
    [cdet, cnum] = cam(tgtPoses, sc_eval.SimulationTime); 
    [rdet, rnum] = rad(tgtPoses, sc_eval.SimulationTime); 
    detections = [cdet; rdet];
    tracks = tracker(detections, sc_eval.SimulationTime);
    for k = 1:numel(tracks)
        tr = tracks(k);
        if ~tr.IsConfirmed, continue;
        end
        tid = tr.TrackID;
        featVector = featureFromTrackState(tr);  
        histRow = [featVector(1), tr.State(3), featVector(5), featVector(6), 0];
        if ~isKey(trackHist, tid)
            trackHist(tid) = histRow;
            [gtLabel, ok] = associateTrackToActor(tr, sc_eval.Actors, ainfo_eval);
            if ok, trackLabelsGT(tid) = gtLabel; else, trackLabelsGT(tid) = 0; end
        else
            prev = trackHist(tid);
            if size(prev,1) >= 1
                dv = hypot(histRow(3)-prev(end,3), histRow(4)-prev(end,4));
                histRow(5) = dv / cfg.SampleTime;
            end
            trackHist(tid) = [prev; histRow]; 
        end
        xFrame = (featVector(1:6) - mu) ./ max(1e-6, sigma);
        yhat_frame = predict(svmModel, xFrame);
        histMat = trackHist(tid); 
        if size(histMat,1) >= 5
            seqForLSTM = histMat(end-4:end, :)'; 
            yhat_seq = classify(lstmNet, seqForLSTM, 'SequenceLength','longest');
            yhat_seq = double(string(yhat_seq)) - 1; 
            finalPred = yhat_seq;
        else
            finalPred = yhat_frame;
        end
        if isKey(trackLabelsGT, tid)
            ev.true(end+1,1) = trackLabelsGT(tid); 
            ev.pred(end+1,1) = finalPred;          
        end
    end
end
if ~isempty(ev.true)
    Ctrk = confusionmat(ev.true, ev.pred);
    accTrack = sum(diag(Ctrk))/sum(Ctrk(:));
    fprintf('Tracked online classification accuracy: %.3f\n', accTrack);
    disp('Confusion matrix [TN FP; FN TP] = ');
    disp(Ctrk);
else
    warning('No evaluation records collected during tracking run.');
end
disp('--- Robustness tests under degraded sensing ---');
scenariosToTest = {struct('camProb',0.95,'radProb',0.95), ...
                   struct('camProb',0.80,'radProb',0.75), ...
                   struct('camProb',0.60,'radProb',0.50)};
for i=1:numel(scenariosToTest)
    sC = scenariosToTest{i};
    cam.DetectionProbability = sC.camProb;
    rad.DetectionProbability = sC.radProb;
    release(tracker); 
    trackHist     = containers.Map('KeyType','double','ValueType','any');
    trackLabelsGT = containers.Map('KeyType','double','ValueType','double');
    ev2.true = []; ev2.pred = [];
    [sc_eval2, ainfo_eval2, ego_eval2] = spawnScene(templateScenario, 'urban_crossing', 5000+i);
    while sc_eval2.SimulationTime < cfg.SampleTime * cfg.SequenceLen
        advance(sc_eval2, cfg.SampleTime);
        tgtPoses = actorPoses(sc_eval2);
        [cdet, ~] = cam(tgtPoses, sc_eval2.SimulationTime);
        [rdet, ~] = rad(tgtPoses, sc_eval2.SimulationTime);
        tracks = tracker([cdet; rdet], sc_eval2.SimulationTime);
        for k=1:numel(tracks)
            tr = tracks(k); if ~tr.IsConfirmed, continue; end
            tid = tr.TrackID;
            featVector = featureFromTrackState(tr);
            histRow = [featVector(1), tr.State(3), featVector(5), featVector(6), 0];
            if ~isKey(trackHist, tid)
                trackHist(tid) = histRow;
                [gtLabel, ok] = associateTrackToActor(tr, sc_eval2.Actors, ainfo_eval2);
                if ok, trackLabelsGT(tid) = gtLabel; else, trackLabelsGT(tid) = 0; end
            else
                prev = trackHist(tid);
                if size(prev,1) >= 1
                    dv = hypot(histRow(3)-prev(end,3), histRow(4)-prev(end,4));
                    histRow(5) = dv / cfg.SampleTime;
                end
                trackHist(tid) = [prev; histRow]; 
            end
            xFrame = (featVector(1:6) - mu) ./ max(1e-6, sigma);
            yhat_frame = predict(svmModel, xFrame);

            histMat = trackHist(tid);
            if size(histMat,1) >= 5
                seqForLSTM = histMat(end-4:end, :)';
                yhat_seq = classify(lstmNet, seqForLSTM, 'SequenceLength','longest');
                yhat_seq = double(string(yhat_seq)) - 1;
                finalPred = yhat_seq;
            else
                finalPred = yhat_frame;
            end

            if isKey(trackLabelsGT, tid)
                ev2.true(end+1,1) = trackLabelsGT(tid); 
                ev2.pred(end+1,1) = finalPred;          
            end
        end
    end
    if ~isempty(ev2.true)
        C2 = confusionmat(ev2.true, ev2.pred);
        acc2 = sum(diag(C2))/sum(C2(:));
        fprintf('cam=%.2f rad=%.2f -> tracked classification acc=%.3f\n', sC.camProb, sC.radProb, acc2);
    else
        fprintf('No eval records for cam=%.2f rad=%.2f\n', sC.camProb, sC.radProb);
    end
end
disp('--- RL environment + agent (simple demo) ---');
obsInfo = rlNumericSpec([3 1]); obsInfo.Name = 'behaviour_obs';
actInfo = rlFiniteSetSpec([1 2 3 4]); actInfo.Name = 'high_level_action';
env = rlFunctionEnv(obsInfo, actInfo, @rlStepFunc, @rlResetFunc);
statePath = [
    featureInputLayer(obsInfo.Dimension(1),'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(24,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(numel(actInfo.Elements),'Name','out')
    ];
critic = rlQValueFunction(rlRepresentation(statePath, obsInfo, actInfo, 'Observation',{'state'}));
agentOpts = rlDQNAgentOptions( ...
    'SampleTime', cfg.SampleTime, ...
    'UseDoubleDQN', false, ...
    'TargetUpdateFrequency', 4, ...
    'ExperienceBufferLength', 1e4, ...
    'MiniBatchSize', 64);
agent = rlDQNAgent(critic, agentOpts);
trainOpts = rlTrainingOptions( ...
    'MaxEpisodes', 60, ...
    'MaxStepsPerEpisode', 200, ...
    'ScoreAveragingWindowLength', 5, ...
    'Verbose', false, ...
    'Plots', 'none');
trainingStats = train(agent, env, trainOpts); 
disp('--- RL agent training complete ---');
function sc = createTemplateScenario(stopTime, sampleTime)
    sc = drivingScenario('StopTime', stopTime, 'SampleTime', sampleTime);
    centers = [0 0 0; 200 0 0];
    road(sc, centers, 'Lanes', lanespec([2 2]));
    road(sc, [80 8 0; 80 -8 0], 'Lanes', lanespec(2));
end
function [scenarioOut, actorInfo, ego] = spawnScene(template, ~, seed)
    rng(seed);
    scenarioOut = createTemplateScenario(template.StopTime, template.SampleTime);
    actorInfo = struct('Actors',[],'IsRisky',[]);
    ego = vehicle(scenarioOut,'ClassID',1,'Position',[10 -1 0],'Length',4.7);
    trajectory(ego,[10 -1 0; 200 -1 0], 25);
    nVeh = randi([2,4]);
    for i=1:nVeh
        xstart = 20 + 30*i + 5*randn;
        lateral = -2 + 4*rand;
        speed   = 18 + 10*rand;
        isRisky = rand < 0.35;
        v = vehicle(scenarioOut,'ClassID',1);
        if isRisky
            xs = [xstart lateral+4 0; xstart+80 lateral-4 0];
            trajectory(v, xs, speed + 8);
        else
            xs = [xstart lateral 0; xstart+120 lateral 0];
            trajectory(v, xs, speed);
        end
        actorInfo.Actors(end+1) = v;
        actorInfo.IsRisky(end+1) = isRisky;
    end
    nPed = randi([1,3]);
    for i=1:nPed
        x = 70 + 10*i + 2*randn;
        y = 6 + randn*0.5;
        ped = actor(scenarioOut,'ClassID',4,'Length',0.5,'Width',0.5,'Height',1.7);
        if rand < 0.3
            traj = [x y 0; x (y-10) 0]; spd = 1.5 + rand*2.5; isRisky = true;
        else
            traj = [x y 0; x (y-10) 0]; spd = 1.0 + rand*0.6;  isRisky = false;
        end
        trajectory(ped, traj, spd);
        actorInfo.Actors(end+1) = ped;
        actorInfo.IsRisky(end+1) = isRisky;
    end
end

function filter = initConstVel2D(detection, Ts)
    dt = Ts;
    F = [1 dt 0  0; 0 1 0 0; 0 0 1 dt; 0 0 0 1];
    H = [1 0 0 0; 0 0 1 0];
    Q = diag([1 1 1 1]) * 0.1;
    R = eye(2) * 4;
    pos = detection.Measurement(1:2);
    X0 = [pos(1); 0; pos(2); 0];
    filter = trackingKF('MotionModel','Custom', ...
        'StateTransitionModel',F, ...
        'MeasurementModel',H, ...
        'State',X0, ...
        'StateCovariance',eye(4)*5, ...
        'ProcessNoise',Q, ...
        'MeasurementNoise',R);
end

function feat = featureFromTrackState(tr)
    st = tr.State; 
    speed = hypot(st(2), st(4));
    vx = st(2); vy = st(4);
    feat = [speed, 0, 0, 0, vx, vy];
end

function [labelGT, ok] = associateTrackToActor(tr, actors, ainfo)
    ok = false; labelGT = 0;
    posT = [tr.State(1) tr.State(3) 0];
    bestD = inf; bestIdx = -1;
    for ii = 1:numel(actors)
        d = norm(posT - actors(ii).Position);
        if d < bestD
            bestD = d; bestIdx = ii;
        end
    end
    if bestD < 8  
        ok = true;
        if bestIdx <= numel(ainfo.IsRisky)
            labelGT = double(ainfo.IsRisky(bestIdx));
        else
            labelGT = 0;
        end
    end
end
function [nextObs, reward, isDone, loggedSignals] = rlStepFunc(env, action, loggedSignals) %#ok<INUSL>
    nextObs = [randi([0 2]); randi([0 2]); randi([0 1])];
    if sum(nextObs) > 0
        if action == 1     
            reward = -1.0;
        else               
            reward = 0.5;
        end
    else
        reward = 0.1;
    end
    isDone = false;
end
function [initialObservation, loggedSignals] = rlResetFunc()
    initialObservation = [0; 0; 0];
    loggedSignals = [];
end
