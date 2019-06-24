function fft_note = fft_note()

som1 = wavread('acordes/sol.wav')
som2 = wavread('acordes/sol_5casa.wav')
som3 = wavread('acordes/sol_pestana_3casa.wav')
som4 = wavread('acordes/sol_pestana_10casa.wav')
som5 = wavread('acordes/sol_quinta.wav')


exec get_mono_signal.sci;
exec get_fourier_transform.sci;

//Get one dimensional array of signal
som1 = get_mono_signal(som1);
som2 = get_mono_signal(som2);
som3 = get_mono_signal(som3);
som4 = get_mono_signal(som4);
som5 = get_mono_signal(som5);

//Get array with slots that are frequencies
respfreq1 = get_fourier_transform(som1);
respfreq2 = get_fourier_transform(som2);
respfreq3 = get_fourier_transform(som3);
respfreq4 = get_fourier_transform(som4);
respfreq5 = get_fourier_transform(som5);

//scf(1);
plot2d([respfreq1, respfreq2, respfreq3, respfreq4, respfreq5]);
e = gce();
hl = captions(e.children, ['sol'; 'sol_5casa'; 'sol_pestana_3casa'; 'sol_pestana_10casa'; 'sol_quinta')

hl.leggend_location='in_upper_right';
hl.fill_mode='on';
mtlb_axis([1, 1000, 0, 1.1]);

endfunction
