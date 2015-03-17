function [notes_time_44100, chords_time_44100, chord_pitch_44100] = DA3(signal, fs)
    [notes_time_44100, chords_time_44100, chord_pitch_44100, chord_pitch_number] = DA3_window_44100(signal, fs);
    %[notes_time_10000, chords_time_10000] = DA3_window_10000(signal, fs, chord_pitch_number);
    %chords_time_44100(1:10)
    %chords_time_10000(1:40)

end

function [notes_time, chords_time, chord_pitch, chord_pitch_number] = DA3_window_44100(signal, fs)
    dictionary_chords = { 'C M', 'C m', 'C aum', 'C dim', ...
     'C# M', 'C# m', 'C# aum', 'C# dim', 'D M', 'D m', 'D aum', 'D dim', ...
     'Eb M', 'Eb m', 'Eb aum', 'Eb dim', 'E M', 'E m', 'E aum', 'E dim', ...
     'F M', 'F m', 'F aum', 'F dim', 'F# M', 'F# m', 'F# aum', 'F# dim', ...
     'G M', 'G m', 'G aum', 'G dim', 'G# M', 'G# m', 'G# aum', 'G# dim', ...
     'A M', 'A m', 'A aum', 'A dim', 'Bb M', 'Bb m', 'Bb aum', 'Bb dim', ...
     'B M', 'B m', 'B aum', 'B dim' };

    % get total seconds of time to mensure the length of music 
    signal = signal(:,1);
    time_seconds_total = fix((length(signal)/fs)); 
    
    % preparing struct to allocate notes in time
    set_of_notes_time = {};
    for set = 1:5
        notes_time(time_seconds_total, 60) = 0;
        set_of_notes_time{set} = notes_time;
    end

    % allocate cell to put the chords throughout music 
    chords_time = {};

    for time = 1:time_seconds_total
        % building a window to short fft
        set_of_windows_signals = build_window_short_fft(signal, time, fs);

        % get frequency spectrum
        set_of_spectrums = get_frequency_spectrum(set_of_windows_signals, fs);

        % get energy of notes
        set_of_notes_time = get_energy_notes(set_of_spectrums, set_of_notes_time, time);
    end


    % binarize set of notes
    for set = 1:5
        notes_time = set_of_notes_time{set};
        
        for time = 1:time_seconds_total
            for note = 1:60
                if notes_time(time, note) < max(max(notes_time))/180
                    notes_time(time, note) = 0;
                else
                    notes_time(time, note) = 1;
                end
            end
        end

        set_of_notes_time{set} = notes_time;
    end

    % get chord in pitch
    [chord_pitch, chord_pitch_number] = get_chord_pitch(set_of_notes_time{1}, time_seconds_total, dictionary_chords);


    set_of_chords_time = get_set_of_chords_time(set_of_notes_time);

    for set = 1:5
        chords_time = set_of_chords_time{set};
        for time = 1:time_seconds_total
            disp(dictionary_chords(chords_time(time)));
        end
        disp('new set');
    end

    
end