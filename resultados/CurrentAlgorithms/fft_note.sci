function fft_note = fft_note(file)

som = wavread(file);

exec get_mono_signal.sci;
exec get_fourier_transform.sci;

//Get one dimensional array of signal
som = get_mono_signal(som);

//Get array with slots that are frequencies
respfreq = get_fourier_transform(som);

scf(1);
plot(respfreq);
xtitle('Resposta em Frequência ' + file, 'Frequência (Hz)', 'Resposta');
mtlb_axis([1, 1000, 0, 1.1]);

endfunction
