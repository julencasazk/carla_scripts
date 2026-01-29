% Build multi-experiment iddata objects for brake->decel and throttle->accel.
%
% Brake CSVs (from brake_fitting_data.py, latest version):
% - Each pulse_index is one experiment that contains:
%   - pre-step: brake=0, throttle=0 for ~0.5 s (logged)
%   - step: brake>0 (random within bin), throttle=0 (logged)
% - We build: u = brake (absolute), y = decel_delta = -(accel_x - mean(accel_x_pre_step))
%
% Throttle CSVs:
% - Each pulse_index is one experiment (currently logs only during throttle pulse).

Ts = 0.01;

% ---- Brake bins (one CSV = one bin) ----
brakeFiles = dir(fullfile("substepping","*.csv"));  % e.g. brake_20kmh_000_005.csv
for k = 1:numel(brakeFiles)
    csvPath = fullfile(brakeFiles(k).folder, brakeFiles(k).name);
    Z = local_make_brake_iddata(csvPath, Ts);
    varName = matlab.lang.makeValidName("Zb_" + erase(brakeFiles(k).name, ".csv"));
    assignin("base", varName, Z);
end

% ---- Throttle bins (one CSV per speed) ----
thrFiles = dir(fullfile("throttle_acel","*.csv"));
for k = 1:numel(thrFiles)
    csvPath = fullfile(thrFiles(k).folder, thrFiles(k).name);
    Z = local_make_iddata_simple(csvPath, Ts, "throttle_delta", "accel_x");
    varName = matlab.lang.makeValidName("Zt_" + erase(thrFiles(k).name, ".csv"));
    assignin("base", varName, Z);
end

disp("Done. Workspace now contains Zb_* and Zt_* iddata variables.");

% -------- local functions --------
function Z = local_make_brake_iddata(csvPath, Ts)
    T = readtable(csvPath);

    required = ["pulse_index", "brake", "accel_x"];
    for c = required
        if ~ismember(c, string(T.Properties.VariableNames))
            error("CSV %s missing required column: %s", csvPath, c);
        end
    end

    pulseIds = unique(T.pulse_index);
    Z = [];

    minStepBrake = 1e-4;
    for i = 1:numel(pulseIds)
        seg = T(T.pulse_index==pulseIds(i), :);
        if height(seg) < 3
            continue;
        end

        u = seg.brake;
        a = seg.accel_x;

        % Find step index (first sample where brake becomes > 0).
        idxStep = find(u > minStepBrake, 1, "first");
        if isempty(idxStep) || idxStep < 2
            % Fallback: use first few samples as baseline if step not found.
            idxBaseline = 1:min(10, numel(a));
        else
            idxBaseline = 1:(idxStep-1);
        end

        aBase = mean(a(idxBaseline));
        y = -(a - aBase); % decel_delta (positive for braking)

        z = iddata(y, u, Ts, ...
            "InputName", "brake", ...
            "OutputName", "decel_delta");

        if isempty(Z)
            Z = z;
        else
            Z = merge(Z, z);
        end
    end

    if isempty(Z)
        error("No valid pulse segments found in %s", csvPath);
    end
end


function Z = local_make_iddata_simple(csvPath, Ts, inputCol, outputCol)
    T = readtable(csvPath);

    % expected columns include: pulse_index, inputCol, outputCol
    pulseIds = unique(T.pulse_index);
    Z = [];

    for i = 1:numel(pulseIds)
        seg = T(T.pulse_index==pulseIds(i), :);

        u = seg.(inputCol);
        y = seg.(outputCol);

        % incremental output (remove per-pulse offset)
        y = y - y(1);

        z = iddata(y, u, Ts, ...
            "InputName", inputCol, ...
            "OutputName", "accel_delta");

        if isempty(Z)
            Z = z;
        else
            Z = merge(Z, z);
        end
    end
end
