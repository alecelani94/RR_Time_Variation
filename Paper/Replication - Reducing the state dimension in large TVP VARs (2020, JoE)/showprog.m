function txtlen = showprog( isim, nsims, progstep, txtlen )
% nicely prints the loop progress with percentage

if mod( isim, floor( nsims / progstep ) ) == 0
    if ispc
        for cnt = 1:txtlen
            fprintf( '\b' );
        end
        txtlen = fprintf( '%d iterations completed (%3.1f%%)', isim, isim / nsims * 100 );
    else
        fprintf( '\r%d iterations completed (%3.1f%%)', isim, isim / nsims * 100 );
    end      
end
