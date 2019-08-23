exec get_frequencies.sci;

function results = create_dataset(filenames, outfiles)
    concat_row = 1; // concatenation accord to the rows
    concat_col = 2; // concatenation accord to the cols
    filenames_size = size(filenames)
    for i=1:filenames_size(1) // get number of chords
        results = [];
        for j=1:filenames_size(2) // get number of variations for each chord
            result = get_frequencies(filenames(i,j));
            result = cat(concat_row, j, result);
            result = cat(concat_row, result, i);
            results = cat(concat_row,  results, result');
        end
        csvWrite(results, outfiles(i));
        disp("dataset writed", outfiles(i));
    end
endfunction
