# set term pdfcairo font ",12" linewidth 2
set term eps font ",12"

set style data histogram
set style histogram cluster gap 1
set style line 1 lc rgb "#0080f0"
set style line 2 lc rgb "#ff0000"
set style line 3 lc rgb "#00c000"
set style line 4 lc rgb "#ffc000"
set style line 5 lc rgb "#f000f0"
set style fill solid noborder

set title "Extrae/3.4.1 overhead"

set datafile separator ";"

set xlabel "Execution test"
set xtics rotate by 45 right font ",9"
set key samplen 1 font ",9" spacing 0.85 left reverse Left
set ylabel "Best time (ns) from 10 runs"
set auto x
set yrange [1:*]
plot \
'overheads.dat' using 2:xtic(1) title col ls 1, \
'' using 3:xtic(1) title col ls 2, \
'' using 4:xtic(1) title col ls 3, \
'' using 5:xtic(1) title col ls 4, \
'' using 6:xtic(1) title col ls 5;
