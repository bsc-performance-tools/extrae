#!/usr/bin/perl

my $CSVSuffix = ".clustered.csv";
my $PlotSuffix = ".cl_*.gnuplot";
my $CPIGeneralSuffix = ".cpistack_general.gnuplot";
my $CPIDetailSuffix = ".cpistack_detail.gnuplot";
my $TAG_MIN = "MIN";
my $TAG_MAX = "MAX";
my $TRUE = 1;
my $FALSE = 0;
my $CID_OFFSET = 4;

my %Renaming = ();

sub calc_Renaming
{
	my ($RenamingTableFile) = @_;

	open RENAMING, "$RenamingTableFile";

	while (defined ($line = <RENAMING>))
	{
		chomp $line;
		$nsteps = @steps = split (/,/, $line);

		for ($i=0; $i<$nsteps; $i++)
		{
			$cur_step = $i + 1;
			$Renaming{$cur_step}{$steps[$i]} = $steps[0];
			print "Renaming{".$cur_step."}{".$steps[$i]."} = ".$steps[0]."\n";
		}
	}

	close RENAMING;
}

sub query_Renaming
{
	my ($Step, $Cluster) = @_;

	if (defined $Renaming{$Step}{$Cluster})
	{
		return $Renaming{$Step}{$Cluster};
	}
	else
	{
		return $Cluster;
	}
}

sub do_Renaming
{
	my ($File, $Step) = @_;

print "zzzzzzzzz $File $Step\n";

    open CSV, "$File";
    open CSV_RENAMED, ">$File.renamed";

    while (defined ($line = <CSV>))
    {
        chomp $line;

		$ntokens = @tokens = split (/,/, $line);

	 	$cid = $tokens[($ntokens-1)];

		if ($cid >= 5)
		{
			$cid -= $CID_OFFSET;
			$cid = query_Renaming ( $Step, $cid );
			$cid += $CID_OFFSET;
			$tokens[($ntokens-1)] = $cid;
		}

		$line = join (",", @tokens);

		print CSV_RENAMED "$line\n";

	}
    close CSV;
    close CSV_RENAMED;
}

my %DimLimits = ();

sub calc_Dimensions
{
	my ($CSVFile) = @_;

	open CSV, "$CSVFile";
	
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

            for (my $i=0; $i<$ndims; $i++)
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
	close CSV;

}

sub print_Dimensions
{
	my $dim;

	foreach $dim (keys %DimLimits)
	{
	    print "Dimension '$dim' ($DimLimits{$dim}{$TAG_MIN}, $DimLimits{$dim}{$TAG_MAX})\n";
	}
}

sub check_Dimension
{
	my ($dim) = @_;
	if (not defined $DimLimits{$dim})
	{
		return $FALSE;
	}
	else
	{
		return $TRUE;
	}
}

sub get_Dimension
{
	my ($dim, $limit) = @_;
	return $DimLimits{$dim}{$limit};
}

my $ARGC = @ARGV;

if ($ARGC == 0)
{
    print "Syntax: animate.pl [-r <renaming_table>] <plots_prefix>\n";
    exit;
}
if ($ARGV[0] eq "-r")
{
    $RenamingEnabled = $TRUE;
    $RenamingTable = $ARGV[1];
	$FirstPlotIndex = 2;
	calc_Renaming($RenamingTable);
}
else
{
	$FirstPlotIndex = 0;
}

my @Plots;
my @CPIGeneral;
my @CPIDetail;

$count = 0;

for (my $i=$FirstPlotIndex; $i<$ARGC; $i++)
{
	$Prefix = $ARGV[$i];

	$nsteps = `ls $Prefix*$CSVSuffix | wc -l`;

	my $j = 0;
	my $k = 0;

	$PrefixStep = $Prefix;
	do
	{
		$CSVFile = $PrefixStep.$CSVSuffix;

		if (-e "$CSVFile")
		{
			$j++;
			$count++;
			calc_Dimensions ($CSVFile);
			if ($RenamingEnabled == $TRUE)
			{
				do_Renaming($CSVFile, $count);
print "CALLED do_Renaming j=$j\n";
			}

			$PlotFile = $PrefixStep.$PlotSuffix;
			$PlotFile = `ls $PlotFile`;
			chomp $PlotFile;
			push (@Plots, $PlotFile);

			$CPIGen = $PrefixStep.$CPIGeneralSuffix;
			if (-e "$CPIGen")
			{
				push (@CPIGeneral, $CPIGen);
			}

			$CPIDet = $PrefixStep.$CPIDetailSuffix;
			if (-e "$CPIDet")
			{
				push (@CPIDetail, $CPIDet);
			}
		}
		$k++;
		$PrefixStep = $Prefix."_".$k;
	}
	while ($j < $nsteps)
}

print_Dimensions();

my $TotalSteps;

$TotalSteps = @Plots;
for (my $Step=0; $Step<$TotalSteps; $Step++)
{
	$PlotFile = $Plots[$Step];

	$nparts = @parts = split (/\./, $PlotFile);

    $DimX = $parts[$nparts - 3];
	(check_Dimension($DimX) == $TRUE) or die "Unexpected dimension X '$DimX'";

    $DimY = $parts[$nparts - 2];
	(check_Dimension($DimY) == $TRUE) or die "Unexpected dimension Y '$DimY'";

	$xrange = "set xrange [".get_Dimension($DimX, $TAG_MIN).":".get_Dimension($DimX, $TAG_MAX)."]";
    $yrange = "set yrange [".get_Dimension($DimY, $TAG_MIN).":".get_Dimension($DimY, $TAG_MAX)."]";

	`echo $xrange >> CLUSTERS_PLOTS.gnuplot`;
	`echo $yrange >> CLUSTERS_PLOTS.gnuplot`;

    if ($RenamingEnabled == $TRUE)
    {
        `cat $PlotFile | sed "s/\.csv/\.csv\.renamed/" >> CLUSTERS_PLOTS.gnuplot`;
    }
    else
    {
        `cat $PlotFile >> CLUSTERS_PLOTS.gnuplot`;
    }
}

$TotalSteps = @CPIGeneral;
for (my $Step=0; $Step<$TotalSteps; $Step++)
{
	`cat $CPIGeneral[$Step] >> CPISTACK_GENERAL.gnuplot`;
}

$TotalSteps = @CPIDetail;
for (my $Step=0; $Step<$TotalSteps; $Step++)
{
	`cat $CPIDetail[$Step] >> CPISTACK_DETAIL.gnuplot`;
}

