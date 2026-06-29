function norm = mel_pipeline_matlab(samples)
% MEL_PIPELINE_MATLAB  MPIC v1.0 — MATLAB reference preprocessing pipeline
%
% Exact port of Python mel_pipeline.py / generate_golden.py for STREAMSENSE.
% Reproduces the Python/torchaudio result to within tolerance 5e-4 (cross-impl).
%
% INPUT
%   samples  : float  [16000 x 1] or [1 x 16000]  — 16 kHz mono PCM, float32
%              May be shorter (zero-padded to 16000) or longer (cropped).
%
% OUTPUT
%   norm     : single [64 x 97]  — column-major, normalised log-mel spectrogram
%              Ready for reshape(norm, [64 97]) in MATLAB (already correct shape).
%              To feed the ONNX model, reshape to [1 1 64 97].
%
% USAGE
%   [samples, fs] = audioread('GV_00_yes.wav');
%   if fs ~= 16000, error('Resample to 16 kHz first'); end
%   norm = mel_pipeline_matlab(samples);           % [64 x 97] single
%   input_tensor = reshape(norm, [1 1 64 97]);     % for ONNX inference
%
% VERIFICATION
%   Load the matching golden vector and compare:
%     fid  = fopen('GV_00_yes_norm.bin', 'rb', 'l');
%     ref  = reshape(fread(fid, 64*97, 'float32=>single'), [64, 97]);
%     fclose(fid);
%     max_err = max(abs(norm(:) - ref(:)));
%     assert(max_err < 5e-4, 'Pipeline mismatch');
%
% MPIC v1.0 FROZEN PARAMETERS (do NOT modify)
%   sample_rate  = 16000 Hz
%   frame_len    = 16000 samples  (1 second)
%   n_fft        = 512
%   hop_length   = 160
%   n_mels       = 64
%   center       = false          ← CRITICAL: gives T=97, not T=98
%   power        = 2.0            (power spectrogram)
%   window       = hann periodic
%   log_scale    = 10 * log10(mel + 1e-10)
%   clip_floor   = -80 dB
%   global_mean  = -30.785545 dB
%   global_std   =  22.157099 dB
%
% Project: STREAMSENSE — Track A
% Spec:    MPIC v1.0 (frozen)

% ── MPIC v1.0 frozen constants ────────────────────────────────────────────────
SAMPLE_RATE   = 16000;
FRAME_LEN     = 16000;
N_FFT         = 512;
HOP_LENGTH    = 160;
N_MELS        = 64;
LOG_EPS       = 1e-10;
CLIP_FLOOR_DB = -80.0;
GLOBAL_MEAN   = -30.785545;   % dB  (from normalization_stats.json)
GLOBAL_STD    =  22.157099;   % dB

% Derived — must equal 97 for MPIC v1.0
EXPECTED_T = (FRAME_LEN - N_FFT) / HOP_LENGTH + 1;  % = 97
EXPECTED_T = floor(EXPECTED_T);   % force integer — MATLAB / gives 97.8 without this

% ── Step 1-3: Pad / crop to exactly FRAME_LEN samples ────────────────────────
samples = single(samples(:));          % force column vector, single precision
L = length(samples);
if L < FRAME_LEN
    samples = [samples; zeros(FRAME_LEN - L, 1, 'single')];
elseif L > FRAME_LEN
    samples = samples(1:FRAME_LEN);
end

% ── Step 4: Build Hann window (periodic, length N_FFT) ───────────────────────
% Python/torchaudio uses a PERIODIC Hann window (N+1 point, drop last).
% MATLAB's hann() uses symmetric by default — must use 'periodic' flag.
win = single(hann(N_FFT, 'periodic'));

% ── Step 5: STFT — center=False means no padding, analyse from sample 1 ──────
% With center=False and hop=160, torchaudio analyses frames starting at:
%   frame k starts at sample  k * HOP_LENGTH  (0-indexed)
% MATLAB's spectrogram() aligns identically when no padding is added.
%
% spectrogram(x, win, noverlap, nfft) with noverlap = N_FFT - HOP_LENGTH
noverlap = N_FFT - HOP_LENGTH;        % = 352

[S, ~, ~] = spectrogram(samples, win, noverlap, N_FFT, SAMPLE_RATE, 'onesided');
% S : [N_FFT/2+1 x T] = [257 x 97]  complex STFT coefficients

% Power spectrogram (power=2.0 → magnitude squared)
power_spec = real(S).^2 + imag(S).^2;   % [257 x 97]  single

% ── Step 6: Mel filterbank — build triangular filters on Hz scale ─────────────
% Mirrors torchaudio's MelSpectrogram filterbank exactly:
%  - freq_min = 0 Hz, freq_max = SAMPLE_RATE/2
%  - N_MELS+2 linearly-spaced points on the mel scale, converted back to Hz
%  - Triangular filters; NO per-filter normalization (norm=None, the default)
%
mel_fmin  = 0.0;
mel_fmax  = SAMPLE_RATE / 2;           % 8000 Hz

% Mel-scale conversion (torchaudio / librosa convention: 2595 * log10(1+f/700))
hz_to_mel = @(f) 2595.0 * log10(1.0 + f / 700.0);
mel_to_hz = @(m) 700.0 * (10.^(m / 2595.0) - 1.0);

mel_min = hz_to_mel(mel_fmin);
mel_max = hz_to_mel(mel_fmax);

% N_MELS+2 equally-spaced mel points → convert to Hz
mel_pts = linspace(mel_min, mel_max, N_MELS + 2);
hz_pts  = mel_to_hz(mel_pts);          % [N_MELS+2] centre-frequencies in Hz

% Map centre-frequencies to STFT bin indices (0-indexed FFT bins → 1-indexed MATLAB)
% STFT bin k corresponds to frequency k * SAMPLE_RATE / N_FFT
n_freqs  = N_FFT / 2 + 1;             % 257
freq_bin = (0:n_freqs-1)' * (SAMPLE_RATE / N_FFT);   % [257 x 1]

% Build filterbank matrix [N_MELS x n_freqs] = [64 x 257]
fb = zeros(N_MELS, n_freqs, 'single');
for m = 1:N_MELS
    f_left   = hz_pts(m);
    f_center = hz_pts(m + 1);
    f_right  = hz_pts(m + 2);

    % Rising slope
    mask_up  = (freq_bin >= f_left)  & (freq_bin <= f_center);
    fb(m, mask_up) = single((freq_bin(mask_up) - f_left) / (f_center - f_left));

    % Falling slope
    mask_dn  = (freq_bin > f_center) & (freq_bin <= f_right);
    fb(m, mask_dn) = single((f_right - freq_bin(mask_dn)) / (f_right - f_center));
end

% IMPORTANT: torchaudio.transforms.MelSpectrogram uses norm=None by default.
% norm=None means NO per-filter bandwidth normalization is applied.
% The 2/bw factor belongs to torchaudio's optional norm='slaney' mode, which
% is NOT the default and NOT what mel_pipeline.py uses.
% Therefore: do NOT apply any normalization here.
% (Applying 2/bw was Bug #1 — it shifted mel dB by 29–52 dB per filter,
%  causing max_err_norm ≈ 1.13 and max_err_mel ≈ 25 dB in verification.)

% ── Step 7: Apply filterbank → mel spectrogram [64 x 97] ─────────────────────
mel_spec = fb * single(power_spec);   % [64 x 257] * [257 x 97] = [64 x 97]

% ── Step 8: Log scale  10 * log10(mel + 1e-10) ───────────────────────────────
mel_db = 10.0 * log10(mel_spec + single(LOG_EPS));

% ── Step 9: Clamp to floor ────────────────────────────────────────────────────
mel_db = max(mel_db, single(CLIP_FLOOR_DB));

% ── Step 10: Global normalisation ────────────────────────────────────────────
norm = (mel_db - single(GLOBAL_MEAN)) / single(GLOBAL_STD);

% norm is [64 x 97] single, column-major in MATLAB memory — correct for the
% golden_vectors_10_matlab/ files which were also saved column-major.

% Validate output shape
if size(norm,1) ~= N_MELS || size(norm,2) ~= EXPECTED_T
    error('mel_pipeline_matlab: unexpected output shape [%d x %d], expected [64 x 97]', ...
          size(norm, 1), size(norm, 2));
end

end % function mel_pipeline_matlab
