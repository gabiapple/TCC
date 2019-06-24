function range_of_frequencies = set_range_of_frequencies(fstart, fend, freq)
    for i=1:length(freq)
        if i < fstart
            cof = round(fstart/i);
            if modulo(cof, 2) == 1
                cof = cof + 1;
            end
            newi = round(i*cof);
            freq(newi) = freq(newi) + freq(i);
            freq(i) = 0;
        elseif i > fend
            cof = round(i/fend);
            if modulo(cof, 2) == 1
                cof = cof + 1;
            end
            newi = round(i/cof);
            freq(newi) = freq(newi) + freq(i);;
            freq(i) = 0;
        end
    end
    range_of_frequencies = freq;
endfunction
