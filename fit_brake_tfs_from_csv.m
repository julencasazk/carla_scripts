% fit_brake_tfs_from_csv.m
%
% Batch-fit discrete-time transfer functions for BRAKE -> ACCEL/DECEL
% from CSV files produced by brake_fitting_data.py.
%
% Assumptions:
% - CSV columns include: pulse_index, brake_delta, accel_x
% - Sampling time is fixed at Ts = 0.01 s
% - Each pulse_index is treated as a separate "experiment" (multi-experiment iddata)
%
% What this script does:
% - For every CSV in BRAKE_CSV_DIR:
%   - Build multi-experiment iddata Z from pulse_index segments
%   - Try candidate TF structures: 1p, 2p, 2p1z, 3p, 3p1z, 3p2z
%   - For each structure, grid-search a discrete input delay (0..MAX_DELAY_SAMPLES)
%   - Reject models with wrong DC gain sign for the chosen output convention
%   - Pick the best by NRMSE fit percent (free simulation vs the same data)
% - Export best model per CSV into the base workspace as sys_<csv_basename>
% - Export a summary table into the base workspace as brake_fit_summary
%
% Output convention (choose one):
%   OUTPUT_MODE = "decel"  => y = -(accel_x - accel_x(1))  (positive when braking)
%   OUTPUT_MODE = "accel"  => y =  (accel_x - accel_x(1))  (negative when braking)
%
% Notes:
% - If all candidates get rejected by sign/stability, the script falls back to the
%   best-fit model even if sign is wrong and marks it in the summary.
%

% ----------------------------
% User settings
% ----------------------------
BRAKE_CSV_DIR = "substepping";
CSV_GLOB = "*.csv";
Ts = 0.01;

OUTPUT_MODE = "decel";           % "decel" or "accel"
MAX_DELAY_SAMPLES = 10;          % 10 samples @ 0.01s => 0.1 s
STEP_TEST_INPUT = 0.1;           % used only for sign logic (DC gain sign check)
VERBOSE = 2;                     % 0: quiet, 1: per-file summary, 2: per-candidate, 3: include tfest errors

% Candidate structures: [npoles, nzeros]
CANDIDATES = [
    1 0
    2 0
    2 1
    3 0
    3 1
    3 2
];

% ----------------------------
% Discover files
% ----------------------------
files = dir(fullfile(BRAKE_CSV_DIR, CSV_GLOB));
if isempty(files)
    error("No CSV files found in %s matching %s", BRAKE_CSV_DIR, CSV_GLOB);
end

fprintf("Found %d brake CSVs in %s\n", numel(files), BRAKE_CSV_DIR);

summaryRows = [];

for k = 1:numel(files)
    csvPath = fullfile(files(k).folder, files(k).name);
    baseName = erase(files(k).name, ".csv");

    fprintf("\n=== [%d/%d] Fitting %s ===\n", k, numel(files), csvPath);

    Z = build_iddata_from_csv(csvPath, Ts, OUTPUT_MODE, VERBOSE);
    [bestSys, bestInfo] = fit_best_tf(Z, CANDIDATES, MAX_DELAY_SAMPLES, STEP_TEST_INPUT, OUTPUT_MODE, VERBOSE);

    varName = matlab.lang.makeValidName("sys_" + baseName);
    assignin("base", varName, bestSys);
    fprintf("Exported %s to base workspace\n", varName);

    row = struct();
    row.csv = string(csvPath);
    row.var = string(varName);
    row.fit_percent = bestInfo.fit_percent;
    row.npoles = bestInfo.npoles;
    row.nzeros = bestInfo.nzeros;
    row.delay_samples = bestInfo.delay_samples;
    row.is_stable = bestInfo.is_stable;
    row.dcgain = bestInfo.dcgain;
    row.sign_ok = bestInfo.sign_ok;
    row.used_fallback = bestInfo.used_fallback;
    summaryRows = [summaryRows; row]; %#ok<AGROW>
end

brake_fit_summary = struct2table(summaryRows);
assignin("base", "brake_fit_summary", brake_fit_summary);
disp("Exported brake_fit_summary to base workspace");


% ----------------------------
% Local functions
% ----------------------------
function Z = build_iddata_from_csv(csvPath, Ts, outputMode, verbose)
    T = readtable(csvPath);

    required = ["pulse_index", "brake_delta", "accel_x"];
    for c = required
        if ~ismember(c, string(T.Properties.VariableNames))
            error("CSV %s missing required column: %s", csvPath, c);
        end
    end

    pulseIds = unique(T.pulse_index);
    Z = [];

    if verbose >= 1
        fprintf("  Loaded %d rows, %d pulses (unique pulse_index)\n", height(T), numel(pulseIds));
        try
            fprintf("  u=brake_delta range: [%.4f, %.4f]\n", min(T.brake_delta), max(T.brake_delta));
        catch
        end
    end

    used = 0;
    segLens = [];
    for i = 1:numel(pulseIds)
        seg = T(T.pulse_index == pulseIds(i), :);
        if height(seg) < 3
            continue;
        end
        used = used + 1;
        segLens(end+1, 1) = height(seg); %#ok<AGROW>

        u = seg.brake_delta;
        a = seg.accel_x;

        % Incremental output per pulse (remove offset).
        a0 = a(1);
        if outputMode == "decel"
            y = -(a - a0);
        else
            y = (a - a0);
        end

        z = iddata(y, u, Ts, "InputName", "brake_delta", "OutputName", "accel_delta");

        if isempty(Z)
            Z = z;
        else
            Z = merge(Z, z);
        end
    end

    if isempty(Z)
        error("No valid pulse segments found in %s", csvPath);
    end

    if verbose >= 1
        fprintf("  Built iddata with %d experiments\n", used);
        if ~isempty(segLens)
            fprintf("  Segment length samples: min=%d max=%d mean=%.1f\n", min(segLens), max(segLens), mean(segLens));
        end
    end
end


function [bestSys, bestInfo] = fit_best_tf(Z, candidates, maxDelay, stepU, outputMode, verbose)
    wantPositive = (outputMode == "decel"); % more brake => more positive decel_delta
    expectedSign = sign(stepU) * (wantPositive * 2 - 1); % +1 for decel, -1 for accel

    best = struct();
    best.score = -inf;
    best.sys = [];
    best.info = [];

    bestAny = best;

    for ci = 1:size(candidates, 1)
        np = candidates(ci, 1);
        nz = candidates(ci, 2);

        if verbose >= 2
            fprintf("  Candidate np=%d nz=%d (delay 0..%d)\n", np, nz, maxDelay);
        end

        for d = 0:maxDelay
            [sys, ok, errMsg] = try_tfest(Z, np, nz, d);
            if ~ok
                if verbose >= 3 && ~isempty(errMsg)
                    fprintf("    tfest FAIL np=%d nz=%d delay=%d: %s\n", np, nz, d, errMsg);
                end
                continue;
            end

            [fitPct, isStable] = sim_fit_percent(Z, sys);
            g = safe_dcgain(sys);
            signOk = (sign(g) == expectedSign) || (abs(g) < 1e-9);

            info = struct();
            info.fit_percent = fitPct;
            info.npoles = np;
            info.nzeros = nz;
            info.delay_samples = d;
            info.is_stable = isStable;
            info.dcgain = g;
            info.sign_ok = signOk;
            info.used_fallback = false;

            if verbose >= 2
                fprintf(
                    "    try np=%d nz=%d delay=%d -> fit=%.2f%% stable=%d dcgain=%.4g sign_ok=%d\n",
                    np, nz, d, fitPct, isStable, g, signOk
                );
            end

            % Keep best model regardless of sign (fallback)
            if fitPct > bestAny.score
                bestAny.score = fitPct;
                bestAny.sys = sys;
                bestAny.info = info;
            end

            % Prefer: stable + correct sign, then fit
            if isStable && signOk
                if fitPct > best.score
                    best.score = fitPct;
                    best.sys = sys;
                    best.info = info;
                end
            end
        end
    end

    if isempty(best.sys)
        bestSys = bestAny.sys;
        bestInfo = bestAny.info;
        bestInfo.used_fallback = true;
        warning( ...
            'No stable+sign-correct model found; using best-fit fallback (np=%d,nz=%d,delay=%d,fit=%.2f%%,dcgain=%.4g).', ...
            bestInfo.npoles, bestInfo.nzeros, bestInfo.delay_samples, bestInfo.fit_percent, bestInfo.dcgain ...
        );
    else
        bestSys = best.sys;
        bestInfo = best.info;
    end

    fprintf( ...
        'Best: np=%d nz=%d delay=%d fit=%.2f%% stable=%d dcgain=%.4g sign_ok=%d fallback=%d\n', ...
        bestInfo.npoles, bestInfo.nzeros, bestInfo.delay_samples, bestInfo.fit_percent, bestInfo.is_stable, bestInfo.dcgain, bestInfo.sign_ok, bestInfo.used_fallback ...
    );
end


function [sys, ok, errMsg] = try_tfest(Z, np, nz, delaySamples)
    ok = false;
    sys = [];
    errMsg = "";

    % Try to set reasonable options if supported.
    opts = [];
    try
        opts = tfestOptions;
        try
            opts.EnforceStability = true;
        catch
        end
        try
            opts.Display = "off";
        catch
        end
    catch
        opts = [];
    end

    % Try multiple call signatures (MATLAB versions differ).
    try
        if ~isempty(opts)
            sys = tfest(Z, np, nz, "InputDelay", delaySamples, opts);
        else
            sys = tfest(Z, np, nz, "InputDelay", delaySamples);
        end
        ok = true;
        return;
    catch ME
        errMsg = string(ME.message);
    end

    try
        if ~isempty(opts)
            sys = tfest(Z, np, nz, delaySamples, opts);
        else
            sys = tfest(Z, np, nz, delaySamples);
        end
        ok = true;
        return;
    catch ME
        errMsg = string(ME.message);
    end
end


function [fitPct, isStable] = sim_fit_percent(Z, sys)
    % Free simulation fit metric: NRMSE across all experiments.
    fitPct = -inf;
    isStable = true;

    try
        isStable = isstable(sys);
    catch
        % best-effort stability check by poles if available
        try
            p = pole(sys);
            isStable = all(abs(p) < 1.0);
        catch
            isStable = true;
        end
    end

    try
        Yhat = sim(sys, Z);
    catch
        try
            % compare sometimes works when sim doesn't
            Yhat = compare(Z, sys);
        catch
            fitPct = -inf;
            return;
        end
    end

    % Extract experiment data (cell arrays for multi-experiment iddata)
    try
        yCell = Z.OutputData;
        yhatCell = Yhat.OutputData;
    catch
        fitPct = -inf;
        return;
    end

    if ~iscell(yCell)
        yCell = {yCell};
    end
    if ~iscell(yhatCell)
        yhatCell = {yhatCell};
    end

    num = 0.0;
    den = 0.0;
    for i = 1:numel(yCell)
        y = yCell{i};
        yhat = yhatCell{i};
        if isempty(y) || isempty(yhat)
            continue;
        end
        y = y(:);
        yhat = yhat(:);
        n = min(numel(y), numel(yhat));
        y = y(1:n);
        yhat = yhat(1:n);
        ym = mean(y);
        num = num + norm(y - yhat);
        den = den + norm(y - ym);
    end

    if den < 1e-12
        fitPct = -inf;
        return;
    end

    fitPct = 100.0 * (1.0 - (num / den));
end


function g = safe_dcgain(sys)
    g = NaN;
    try
        g = dcgain(sys);
        if numel(g) > 1
            g = g(1);
        end
    catch
        g = NaN;
    end
end
