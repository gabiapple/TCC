exec get_mono_signal.sci;
exec get_fourier_transform.sci;
exec get_max_values_range.sci;


function results = get_frequencies(filename)
    [som, fs] = wavread(filename);
    som = get_mono_signal(som);
    respfreq = get_fourier_transform(som, fs);
    [max1, max2, max2val] = get_max_values_range(respfreq, 100, 65, 1000);
    results = max2;
endfunction
