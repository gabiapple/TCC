function get_max_values(data, step, N)
    j=1;
    for i=1:step:N
        [maximum, index] = max(fourier(i:i+step-1));
        disp(index);
    end
endfunction
