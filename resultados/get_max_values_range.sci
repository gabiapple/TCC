function [maximus, maximus2, maximus2val] = get_max_values_range(data, step, fstart, N)
    j=1;
    k=1;
    start_f = 125;
    end_f = 250;
    for i=fstart:step:N
        [maximum, index] = max(data(i:i+step-1));
        f = index + i - 1;
         if f < start_f
            cof = round(start_f/i);
            if modulo(cof, 2) == 1
                cof = cof + 1;
            end
            newf = round(f*cof);
            maximus2(k) = f;
            maximus2val(k) = data(f);
            k = k + 1;
        elseif i > end_f
            cof = round(i/end_f);
            if modulo(cof, 2) == 1
                cof = cof + 1;
            end
            newf = round(f/cof);
            maximus2(k) = newf;
            maximus2val(k) = data(f);
            k = k + 1;
        end
        maximus(j) = f;
        j = j + 1;
    end
endfunction
