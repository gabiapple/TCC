function [maximus, maximus2, maximus2val] = get_max_values_range(data, step, fstart, N)
    j=1;
    k=1;
    start_f = 125;
    end_f = 255;
    for i=fstart:step:N-step
        [maximum, index] = max(data(i:i+step-1));
        f = index + i - 1;
        newf = f;
        while newf < start_f
            newf = newf * 2;
        end
        while newf > end_f
            newf = newf / 2;
        end
        newf = int(newf);
        f = int(f);
        maximus2(k) = newf;
        maximus2val(k) = data(f);
        maximus(j) = f;
        k = k + 1;
        j = j + 1;
    end
endfunction
