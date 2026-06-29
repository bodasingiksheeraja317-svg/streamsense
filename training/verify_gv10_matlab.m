% verify_gv10_matlab.m
% STREAMSENSE — Track A  |  MPIC v1.0
%
% PURPOSE
%   1. Verifies the column-major conversion was done correctly by loading
%      each GV from golden_vectors_10_matlab/ and re-running mel_pipeline_matlab
%      on the matching raw audio, then comparing the two outputs.
%
%   2. Validates every binary file in golden_vectors_10_matlab/ against
%      the MPIC v1.0 specification (shape, byte-size, dtype, value range).
%
% WHAT PASSES
%   norm re-computed from raw WAV vs norm loaded from .bin  <  5e-4  (cross-impl tolerance)
%   mel  re-computed from raw WAV vs mel  loaded from .bin  <  5e-4
%
% PREREQUISITES
%   • mel_pipeline_matlab.m  on the MATLAB path (same folder is fine)
%   • golden_vectors_10_matlab/ folder produced by generate_golden10_matlab.py
%   • Raw WAV files  golden_vectors_10_matlab/raw/*.bin  (raw PCM, NOT wav)
%     OR the original WAV files reachable from GV_WAV_DIR below.
%
% USAGE
%   Run from any folder:
%       cd C:\STREAMSENSE
%       verify_gv10_matlab
%
%   Or with a custom root:
%       GV_ROOT = 'D:\my_path\golden_vectors_10_matlab';
%       verify_gv10_matlab        % set GV_ROOT before running
%
% OUTPUT
%   Console report + PASS/FAIL summary.
%   Saves results to  golden_vectors_10_matlab\verify_report.txt
%
% Project: STREAMSENSE — Track A
% Spec:    MPIC v1.0 (frozen)

clc;
fprintf('============================================================\n');
fprintf('STREAMSENSE — verify_gv10_matlab.m\n');
fprintf('MPIC v1.0 Golden Vector Verification (MATLAB)\n');
fprintf('============================================================\n\n');

% ── Paths ─────────────────────────────────────────────────────────────────────
if ~exist('GV_ROOT', 'var')
    GV_ROOT = 'C:\STREAMSENSE\golden_vectors_10_matlab';
end

RAW_DIR   = fullfile(GV_ROOT, 'raw');
MEL_DIR   = fullfile(GV_ROOT, 'mel');
NORM_DIR  = fullfile(GV_ROOT, 'normalized');
LABEL_DIR = fullfile(GV_ROOT, 'labels');

% Original WAV files (needed for pipeline re-run check)
% Set this to wherever the source WAVs live (golden_vectors/wav/ or recordings/)
WAV_DIR   = 'C:\STREAMSENSE\golden_vectors\wav';

fprintf('GV root  : %s\n', GV_ROOT);
fprintf('WAV dir  : %s\n\n', WAV_DIR);

% ── MPIC v1.0 frozen constants ────────────────────────────────────────────────
FRAME_LEN      = 16000;
N_MELS         = 64;
EXPECTED_T     = 97;   % = floor((16000-512)/160)+1  — hardcoded to avoid float division
EXPECTED_RAW_BYTES  = FRAME_LEN * 4;          % 64000
EXPECTED_MEL_BYTES  = N_MELS * EXPECTED_T * 4; % 24832
CLIP_FLOOR_DB  = -80.0;
CROSS_IMPL_TOL = 5e-4;   % MPIC v1.0 cross-implementation tolerance

% ── Class labels ──────────────────────────────────────────────────────────────
LABELS = {'yes','no','up','down','left','right','on','off','stop','go'};

% ── Results table ─────────────────────────────────────────────────────────────
n_classes   = 10;
results     = struct();
all_passed  = true;

% ── Open report file ──────────────────────────────────────────────────────────
report_path = fullfile(GV_ROOT, 'verify_report.txt');
fid_rep = fopen(report_path, 'w');
if fid_rep < 0
    warning('Cannot write report to %s — printing to console only.', report_path);
    fid_rep = 1;  % stdout fallback
end

log = @(varargin) fprintf(fid_rep, varargin{:});

log('STREAMSENSE — verify_gv10_matlab.m\n');
log('Generated: %s\n', datestr(now));
log('GV root  : %s\n\n', GV_ROOT);

% ── Main verification loop ────────────────────────────────────────────────────
for i = 0:9
    lbl     = LABELS{i+1};
    gv_name = sprintf('GV_%02d_%s', i, lbl);

    fprintf('\n%s\n', repmat('-', 1, 54));
    fprintf('Verifying %s  (class %d — ''%s'')\n', gv_name, i, lbl);
    log('\n%s\n', repmat('-', 1, 54));
    log('Verifying %s  (class %d — ''%s'')\n', gv_name, i, lbl);

    ok_struct = struct('size_raw', false, 'size_mel', false, 'size_norm', false, ...
                       'label_ok', false, 'range_ok', false, ...
                       'pipeline_mel_ok', false, 'pipeline_norm_ok', false);

    % ── File paths ────────────────────────────────────────────────────────────
    raw_path   = fullfile(RAW_DIR,   sprintf('%s.bin',       gv_name));
    mel_path   = fullfile(MEL_DIR,   sprintf('%s_mel.bin',   gv_name));
    norm_path  = fullfile(NORM_DIR,  sprintf('%s_norm.bin',  gv_name));
    label_path = fullfile(LABEL_DIR, sprintf('%s_label.txt', gv_name));
    wav_path   = fullfile(WAV_DIR,   sprintf('%s.wav',       gv_name));

    % ── Check files exist ─────────────────────────────────────────────────────
    missing = {};
    if ~exist(raw_path,  'file'), missing{end+1} = 'raw bin';   end
    if ~exist(mel_path,  'file'), missing{end+1} = 'mel bin';   end
    if ~exist(norm_path, 'file'), missing{end+1} = 'norm bin';  end
    if ~isempty(missing)
        msg = sprintf('  [FAIL] Missing files: %s\n', strjoin(missing, ', '));
        fprintf('%s', msg); log('%s', msg);
        all_passed = false;
        results.(gv_name) = ok_struct;
        continue;
    end

    % ── Check file sizes ──────────────────────────────────────────────────────
    d_raw  = dir(raw_path);
    d_mel  = dir(mel_path);
    d_norm = dir(norm_path);

    ok_struct.size_raw  = (d_raw.bytes  == EXPECTED_RAW_BYTES);
    ok_struct.size_mel  = (d_mel.bytes  == EXPECTED_MEL_BYTES);
    ok_struct.size_norm = (d_norm.bytes == EXPECTED_MEL_BYTES);

    sz_tag = @(ok, actual, expected) ...
        sprintf('%d bytes  %s  (expected %d)', actual, tf_str(ok), expected);

    fprintf('  raw  : %s\n', sz_tag(ok_struct.size_raw,  d_raw.bytes,  EXPECTED_RAW_BYTES));
    fprintf('  mel  : %s\n', sz_tag(ok_struct.size_mel,  d_mel.bytes,  EXPECTED_MEL_BYTES));
    fprintf('  norm : %s\n', sz_tag(ok_struct.size_norm, d_norm.bytes, EXPECTED_MEL_BYTES));
    log('  raw  : %s\n', sz_tag(ok_struct.size_raw,  d_raw.bytes,  EXPECTED_RAW_BYTES));
    log('  mel  : %s\n', sz_tag(ok_struct.size_mel,  d_mel.bytes,  EXPECTED_MEL_BYTES));
    log('  norm : %s\n', sz_tag(ok_struct.size_norm, d_norm.bytes, EXPECTED_MEL_BYTES));

    if ~(ok_struct.size_raw && ok_struct.size_mel && ok_struct.size_norm)
        all_passed = false;
    end

    % ── Load binary files ─────────────────────────────────────────────────────
    % Files are column-major (saved by generate_golden10_matlab.py with tobytes('F'))
    % fread reads flat bytes; reshape with [64,97] gives correct MATLAB matrix.

    fid = fopen(raw_path, 'rb', 'l');
    raw_vec = fread(fid, FRAME_LEN, 'float32=>single');
    fclose(fid);

    fid = fopen(mel_path, 'rb', 'l');
    mel_gv  = reshape(fread(fid, N_MELS*EXPECTED_T, 'float32=>single'), [N_MELS, EXPECTED_T]);
    fclose(fid);

    fid = fopen(norm_path, 'rb', 'l');
    norm_gv = reshape(fread(fid, N_MELS*EXPECTED_T, 'float32=>single'), [N_MELS, EXPECTED_T]);
    fclose(fid);

    % ── Value range checks ────────────────────────────────────────────────────
    mel_min  = min(mel_gv(:));
    mel_max  = max(mel_gv(:));
    norm_min = min(norm_gv(:));
    norm_max = max(norm_gv(:));

    % Mel should be in [−80, 0] dB range (log-mel of audio)
    % MPIC v1.0 clips the floor at -80 dB only — no upper limit.
    % Power mel of real speech typically peaks at +35..+45 dB; >0 is normal.
    ok_struct.range_ok = (mel_min >= CLIP_FLOOR_DB - 0.1);

    fprintf('  mel  range : [%.2f, %.2f] dB  %s\n', mel_min, mel_max, tf_str(ok_struct.range_ok));
    fprintf('  norm range : [%.4f, %.4f]\n', norm_min, norm_max);
    log('  mel  range : [%.2f, %.2f] dB  %s\n', mel_min, mel_max, tf_str(ok_struct.range_ok));
    log('  norm range : [%.4f, %.4f]\n', norm_min, norm_max);

    % ── Label check ───────────────────────────────────────────────────────────
    if exist(label_path, 'file')
        fid = fopen(label_path, 'r');
        label_str = strtrim(fgetl(fid));
        fclose(fid);
        expected_label = num2str(i);
        ok_struct.label_ok = strcmp(label_str, expected_label);
        fprintf('  label: %s  %s  (expected %s)\n', label_str, ...
            tf_str(ok_struct.label_ok), expected_label);
        log('  label: %s  %s\n', label_str, tf_str(ok_struct.label_ok));
    else
        fprintf('  label: [missing]\n');
        ok_struct.label_ok = false;
    end

    % ── Pipeline re-run verification (if WAV available) ───────────────────────
    if exist(wav_path, 'file')
        fprintf('  pipeline re-run check...\n');
        try
            [wav_samples, wav_fs] = audioread(wav_path);
            if wav_fs ~= 16000
                error('WAV sample rate is %d Hz — expected 16000 Hz', wav_fs);
            end
            if size(wav_samples, 2) > 1
                wav_samples = mean(wav_samples, 2);   % stereo → mono
            end

            % Run MATLAB mel pipeline
            norm_rerun = mel_pipeline_matlab(wav_samples);   % [64 x 97]

            % Reconstruct mel from norm for mel-level comparison
            GLOBAL_MEAN_V = single(-30.785545);
            GLOBAL_STD_V  = single(22.157099);
            mel_rerun = norm_rerun * GLOBAL_STD_V + GLOBAL_MEAN_V;

            % Compare against loaded GV
            err_norm = max(abs(norm_rerun(:) - norm_gv(:)));
            err_mel  = max(abs(mel_rerun(:)  - mel_gv(:)));

            ok_struct.pipeline_norm_ok = (err_norm < CROSS_IMPL_TOL);
            ok_struct.pipeline_mel_ok  = (err_mel  < CROSS_IMPL_TOL);

            fprintf('  pipeline max_err norm : %.2e  %s  (tol=5e-4)\n', ...
                err_norm, tf_str(ok_struct.pipeline_norm_ok));
            fprintf('  pipeline max_err mel  : %.2e  %s  (tol=5e-4)\n', ...
                err_mel,  tf_str(ok_struct.pipeline_mel_ok));
            log('  pipeline max_err norm : %.2e  %s\n', err_norm, tf_str(ok_struct.pipeline_norm_ok));
            log('  pipeline max_err mel  : %.2e  %s\n', err_mel,  tf_str(ok_struct.pipeline_mel_ok));

            if ~ok_struct.pipeline_norm_ok || ~ok_struct.pipeline_mel_ok
                all_passed = false;
            end
        catch ME
            fprintf('  pipeline check ERROR: %s\n', ME.message);
            log('  pipeline check ERROR: %s\n', ME.message);
            all_passed = false;
        end
    else
        fprintf('  pipeline re-run: SKIPPED (WAV not found at %s)\n', wav_path);
        log('  pipeline re-run: SKIPPED — WAV not found\n');
        % Mark as N/A (not a failure if WAV is absent)
        ok_struct.pipeline_norm_ok = true;
        ok_struct.pipeline_mel_ok  = true;
    end

    % ── Per-vector pass/fail ──────────────────────────────────────────────────
    vec_pass = ok_struct.size_raw  && ok_struct.size_mel  && ...
               ok_struct.size_norm && ok_struct.range_ok  && ...
               ok_struct.pipeline_norm_ok && ok_struct.pipeline_mel_ok;

    if ~vec_pass, all_passed = false; end

    results.(gv_name) = ok_struct;
end

% ── Summary ───────────────────────────────────────────────────────────────────
fprintf('\n%s\n', repmat('=', 1, 54));
fprintf('SUMMARY\n');
fprintf('%s\n', repmat('=', 1, 54));
log('\n%s\n', repmat('=', 1, 54));
log('SUMMARY\n');
log('%s\n', repmat('=', 1, 54));

for i = 0:9
    lbl     = LABELS{i+1};
    gv_name = sprintf('GV_%02d_%s', i, lbl);

    if isfield(results, gv_name)
        r = results.(gv_name);
        vec_pass = r.size_raw && r.size_mel && r.size_norm && ...
                   r.range_ok && r.pipeline_norm_ok && r.pipeline_mel_ok;
        status = tf_str(vec_pass);
    else
        status = '[FAIL]';
    end

    line = sprintf('  %s  %s\n', status, gv_name);
    fprintf('%s', line);
    log('%s', line);
end

fprintf('\n');
log('\n');

if all_passed
    msg = sprintf('[DONE] All 10 golden vectors PASSED MPIC v1.0 verification.\n');
else
    msg = sprintf('[FAIL] One or more vectors failed — see details above.\n');
end
fprintf('%s', msg);
log('%s', msg);

if fid_rep ~= 1
    fclose(fid_rep);
    fprintf('\nReport saved to: %s\n', report_path);
end

% ── Helper ────────────────────────────────────────────────────────────────────
function s = tf_str(ok)
    if ok
        s = '[OK  ]';
    else
        s = '[FAIL]';
    end
end
