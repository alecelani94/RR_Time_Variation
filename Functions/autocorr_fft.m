function acf = autocorr_fft(y, order)

% Demean 'y'
y = y-mean(y);

% Forward transform / inverse transform
nFFT = 2^(nextpow2(length(y))+1);
F    = fft(y, nFFT);
F    = F .* conj(F);
acf  = ifft(F);

% Normalize and return
acf = acf(1:(order+1));
acf = real(acf);
acf = acf./acf(1);
