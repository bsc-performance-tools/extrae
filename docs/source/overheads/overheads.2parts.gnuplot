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

set multiplot title "Extrae/3.3.0 overhead"

set size 1,0.75
set origin 0,-0.15
set border 11

set datafile separator ";"

set grid ytics
set xtics rotate by 45 right font ",9" nomirror
set ytics font ",9" format "%6.0f"
unset key
# set ylabel "Best time (ns) from 10 runs"
set label "Best time (ns) from 10 runs" at -2.5, 1000 center rotate by 90
set y2label " "
set auto x
set yrange [0:1000] 
plot \
'overheads.dat' using 2:xtic(1) title col ls 1, \
'' using 3:xtic(1) title col ls 2, \
'' using 4:xtic(1) title col ls 3, \
'' using 5:xtic(1) title col ls 4, \
'' using 6:xtic(1) title col ls 5;

set size 1,0.45
set origin 0,0.505
set border 14

unset label

set xlabel "Execution test"
unset xlabel
set ytics (5000, 10000, 20000) font ",9" format "%6.0f"
unset xtics
set key samplen 1 font ",9" spacing 0.85 left reverse Left
# set ylabel " "
set auto x
# set log y
set yrange [1000:20000]
plot \
'overheads.dat' using 2:xtic(1) title col ls 1, \
'' using 3:xtic(1) title col ls 2, \
'' using 4:xtic(1) title col ls 3, \
'' using 5:xtic(1) title col ls 4, \
'' using 6:xtic(1) title col ls 5;

unset multiplot
