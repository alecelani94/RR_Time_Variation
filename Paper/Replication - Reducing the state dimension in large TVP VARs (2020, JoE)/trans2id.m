function [s_Axid, s_gammaxid, s_Ahid, s_gammahid] = trans2id( s_Ax, s_gammax, s_Ah, s_gammah, sharedvol )

[nk, rx, ~] = size( s_Ax );
T = size( s_gammax, 2 );
[n, rh, svsims] = size( s_Ah );

s_Axid = zeros( nk, rx, svsims );
s_gammaxid = zeros( rx, T, svsims );
s_Ahid = zeros( n, rh, svsims );
s_gammahid = zeros( rh, T, svsims );

for isim = 1:svsims
    Ax = s_Ax( :, :, isim );
    gammax = s_gammax( :, :, isim );
    Ah = s_Ah( :, :, isim );
    gammah = s_gammah( :, :, isim );
    if sharedvol < 2
        [Ux, Dx] = svd( Ax, 0 );
        Cx = Ax \ ( Ux * Dx * diag( sign( Ux( 1, : ) ) ) );
        Ax = Ax * Cx; gammax = Cx' * gammax;
        if sharedvol == 1
            [Uh, Dh] = svd( Ah, 0 );
            Ch = Ah \ ( Uh * Dh * diag( sign( Uh( 1, : ) ) ) );
            Ah = Ah * Ch; gammah = Ch' * gammah;
        end
    else
        [U, D] = svd( [Ax; Ah], 0 );
        C = [Ax; Ah] \ ( U * D * diag( sign( U( 1, : ) ) ) );
        Ax = Ax * C; gammax = C' * gammax;
        Ah = Ah * C; gammah = gammax;
    end
    
    s_Axid( :, :, isim ) = Ax;
    s_gammax( :, :, isim ) = gammax;
    s_Ahid( :, :, isim ) = Ah;
    s_gammah( :, :, isim ) = gammah;
end