exec dc.sci;

function repetitive_chords_test = repetitive_chords_test(som, niter, id)
    file_results = mopen('repetitive_chords_test.txt','a');
    results = []
    for i = 1:niter
        results($+1) = string(DA2(som));
    end
    mputl('Recognizition test with chord ' + id + ' niter: ' + string(niter), file_results);
    mputl(strcat(results, ";"), file_results);
    disp(results);
    mclose(file_results);
endfunction



sol = wavread('acordes/sol.wav')

sol_5casa = wavread('acordes/sol_5casa.wav')

sol_pestana_3casa = wavread('acordes/sol_pestana_3casa.wav')

sol_pestana_10casa = wavread('acordes/sol_pestana_10casa.wav')

sol_quinta = wavread('acordes/sol_quinta.wav')

niter = 21;

repetitive_chords_test(sol, niter, 'sol');

repetitive_chords_test(sol_5casa, niter, 'sol_5casa');

repetitive_chords_test(sol_pestana_3casa, niter, 'sol_pestana_3casa');

repetitive_chords_test(sol_pestana_10casa, niter, 'sol_pestana_10casa');

repetitive_chords_test(sol_quinta, niter, 'sol_quinta');
