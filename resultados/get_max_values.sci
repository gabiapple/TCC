function maximus = get_max_values(data, step, fstart, N)
    j=1;
    for i=fstart:step:N
        [maximum, index] = max(data(i:i+step-1));
        disp(index + i -1);
        disp(maximum);
        disp('-----------------------');
        maximus(j) = index + i - 1;
        j = j + 1;
    end
maximus = maximus;
endfunction
