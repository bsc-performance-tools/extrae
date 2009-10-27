#!/usr/bin/perl

my $CSVSuffix = ".clustered.csv";
my $PlotSuffix = ".cl_*.gnuplot";
my $CPIGeneralSuffix = ".cpistack_general.gnuplot";
my $CPIDetailSuffix = ".cpistack_detail.gnuplot";
my $TAG_MIN = "MIN";
my $TAG_MAX = "MAX";

my $ARGC = @ARGV;

if ($ARGC != 1) {
	print "Syntax: animate.pl <plots_prefix>\n";
	exit;
}

my $PlotsPrefix = $ARGV[0];

my %DimLimits = ();

$Step = 1;
$CSVFile = $PlotsPrefix."_".$Step.$CSVSuffix;

while (open CSV, "$CSVFile")
{
	my $read_lines = 0;
	while (defined ($line = <CSV>))
	{
		chomp $line;
		if ($read_lines == 0)
		{
			$line =~ s/\s//g;
			$ndims = @dims = split (/,/, $line);
		}
		else
		{
			@vals = split (/,/, $line);
			
			for ($i=0; $i<$ndims; $i++)
			{
				if ((not defined $DimLimits{$dims[$i]}{$TAG_MIN}) or ($vals[$i] < $DimLimits{$dims[$i]}{$TAG_MIN}))
				{
					$DimLimits{$dims[$i]}{$TAG_MIN} = $vals[$i];
				}
				if ((not defined $DimLimits{$dims[$i]}{$TAG_MAX}) or ($vals[$i] > $DimLimits{$dims[$i]}{$TAG_MAX}))
				{
					$DimLimits{$dims[$i]}{$TAG_MAX} = $vals[$i];
				}
			}
		}
		$read_lines ++;
	}

	$Step ++;
	close CSV;
	$CSVFile = $PlotsPrefix."_".$Step.$CSVSuffix;
}

foreach $dim (keys %DimLimits)
{
	print "Dimension '$dim' ($DimLimits{$dim}{$TAG_MIN}, $DimLimits{$dim}{$TAG_MAX})\n";
}

my $TotalSteps = $Step;
$Step = 1;
do 
{
	$PlotTemplate = $PlotsPrefix."_".$Step.$PlotSuffix;
	$PlotFile = `ls $PlotTemplate`;
	chomp $PlotFile;
	if (not open PLOT, "$PlotFile")
	{
		die "Error: File not found: $PlotFile";
	}
	$nparts = @parts = split (/\./, $PlotFile);

	$DimX = $parts[$nparts - 3];
	if (not defined $DimLimits{$DimX}) 
	{
		die "Unexpected dimension X '$DimX'";
	}
	$DimY = $parts[$nparts - 2];
	if (not defined $DimLimits{$DimY}) 
	{
		die "Unexpected dimension Y '$DimY'";
	}

	$xrange = "set xrange [".$DimLimits{$DimX}{$TAG_MIN}.":".$DimLimits{$DimX}{$TAG_MAX}."]";
	$yrange = "set yrange [".$DimLimits{$DimY}{$TAG_MIN}.":".$DimLimits{$DimY}{$TAG_MAX}."]";

	`echo $xrange >> CLUSTERS_PLOTS.gnuplot`;
	`echo $yrange >> CLUSTERS_PLOTS.gnuplot`;
	`cat $PlotFile >> CLUSTERS_PLOTS.gnuplot`;

	$CPIGeneralFile = $PlotsPrefix."_".$Step.$CPIGeneralSuffix;
	`cat $CPIGeneralFile >> CPISTACK_GENERAL.gnuplot`;

	$CPIDetailFile = $PlotsPrefix."_".$Step.$CPIDetailSuffix;
	`cat $CPIDetailFile >> CPISTACK_DETAIL.gnuplot`;

	$Step ++;
	close PLOT;
} while ($Step < $TotalSteps);

