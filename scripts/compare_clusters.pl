#!/usr/bin/perl

### Defines ###
my $FALSE = 0;
my $TRUE  = 1;
my $RANGE_MIN = "RANGE_MIN";
my $RANGE_MAX = "RANGE_MAX";
my $RANGE_VAR = "RANGE_VAR";
my $LINE_MIN  = "LINE_MIN";
my $LINE_MAX  = "LINE_MAX";
my $OTHER_EDGES = "OTHER_EDGES";
my $MAX_VAR_PCT = 5; # Pct
my $MIN_DURATION_PCT = 85;
my $RESERVED_CIDS = 4;

### Global variables ###
my %EdgesVariations = ();

sub min
{
    my ($x, $y) = @_;
	if ((not defined $x) && (not defined $y)) { return 0; }
	elsif (not defined $x) { return $y; }
	elsif (not defined $y) { return $x; }
    else { return ($x < $y ? $x : $y); }
}

sub max {
    my ($x, $y) = @_;
    if ((not defined $x) && (not defined $y)) { return 0; }
    elsif (not defined $x) { return $y; }
    elsif (not defined $y) { return $x; }
    return ($x > $y ? $x : $y);
}

sub PrettyCID
{
    my ($CID) = @_;
    return ($CID - $RESERVED_CIDS);
}

sub LoadRanges 
{
	my ($CSVFile) = @_;
	my $NumDimensions = 0;
	my @Dimensions = ( );
	my $num_line = 1;
	my %Ranges = ( );

	open CSV, "$CSVFile" or die "Error opening file '$CSVFile'\n$!\n";

	while (defined ($line = <CSV>)) 
	{
	    chomp $line;

	    if ($num_line == 1)
	    {
			# Get the names of the dimensions
	        $NumDimensions = @Dimensions = split(/,/, $line);
	    }
	    else
	    {
			# Find the mininum and maximum value for each dimension and cluster
	        $NumValues = @Values = split (/,/, $line);
	        if ($NumValues != ($NumDimensions + 1))
	        {
	            die "Bad syntax";
	        }
	        $ClusterID = @Values[$NumValues - 1];

			# First ClusterID's are for noise, filtered... we have to skip those
			if ($ClusterID > $RESERVED_CIDS)
			{
				$ClusterID = PrettyCID($ClusterID);
	
		        for ($i=0; $i<$NumDimensions; $i++)
		        {
					my $prev_min, $prev_max, $min, $max, $absmin, $absmax, $min_updated, $max_updated;

					$min_updated = $FALSE;
					$max_updated = $FALSE;

					$prev_min = $Ranges{$ClusterID}{$Dimensions[$i]}{$RANGE_MIN};
					$prev_max = $Ranges{$ClusterID}{$Dimensions[$i]}{$RANGE_MAX};

					$min = min($prev_min, $Values[$i]);
					$max = max($prev_max, $Values[$i]);
	            
					# Save the min and max for this dimension and cluster
					if (($min < $prev_min) || (not defined $prev_min))
					{
						$Ranges{$ClusterID}{$Dimensions[$i]}{$RANGE_MIN} = $min;
						$Ranges{$ClusterID}{$Dimensions[$i]}{$LINE_MIN} = $num_line;
						$min_updated = $TRUE;
					}
					if (($max > $prev_max) || (not defined $prev_max))
					{
						$Ranges{$ClusterID}{$Dimensions[$i]}{$RANGE_MAX} = $max;
						$Ranges{$ClusterID}{$Dimensions[$i]}{$LINE_MAX} = $num_line;
						$max_updated = $TRUE;
					}

					# Save the values of the other edges when the currend edge has min or max value
					if (($min_updated) || ($max_updated))
					{
						for ($j=0; $j<$NumDimensions; $j++)
						{
							if ($j != $i)
							{
								if ($min_updated) { $Ranges{$ClusterID}{$Dimensions[$i]}{$OTHER_EDGES}{$Dimensions[$j]}{$RANGE_MIN} = $Values[$j]; }
								if ($max_updated) { $Ranges{$ClusterID}{$Dimensions[$i]}{$OTHER_EDGES}{$Dimensions[$j]}{$RANGE_MAX} = $Values[$j]; }
							}
						}
					}

					# Save the absolute min and max for each dimension
					$absmin = min($min, $EdgesVariations{$Dimensions[$i]}{$RANGE_MIN});
					$absmax = max($max, $EdgesVariations{$Dimensions[$i]}{$RANGE_MAX});

					# This is stored in a global hash so it is shared amongst all clusterings				
					$EdgesVariations{$Dimensions[$i]}{$RANGE_MIN} = $absmin;
					$EdgesVariations{$Dimensions[$i]}{$RANGE_MAX} = $absmax;
	        	}
			}
	    }
		$num_line ++;
	}

	# Compute the maximum variation for each dimension
	foreach $dim (keys %EdgesVariations)
	{
		$EdgesVariations{$dim}{$RANGE_VAR} = ((($EdgesVariations{$dim}{$RANGE_MAX} - $EdgesVariations{$dim}{$RANGE_MIN}) * $MAX_VAR_PCT) / 100);
		print "DIM: $dim\n";
		print "__ RANGE_MIN: $EdgesVariations{$dim}{$RANGE_MIN}\n";
		print "__ RANGE_MAX: $EdgesVariations{$dim}{$RANGE_MAX}\n";
		print "__ RANGE_VAR: $EdgesVariations{$dim}{$RANGE_VAR}\n";
	}

	close CSV or die "Error closing file '$CSVFile'\n$!\n";

	return %Ranges;
}

sub PrintRanges 
{
	my ($hash_ptr) = @_;
	my %Ranges = %{$hash_ptr};

	foreach $ClusterID (sort keys %Ranges)
	{
		print "ClusterID: $ClusterID\n";
		foreach $Dimension (keys %{$Ranges{$ClusterID}})
		{
			my $min, $max;

			$min = $Ranges{$ClusterID}{$Dimension}{$RANGE_MIN};
			$max = $Ranges{$ClusterID}{$Dimension}{$RANGE_MAX};
			print "___ $Dimension ($min, $max) (lines $Ranges{$ClusterID}{$Dimension}{$LINE_MIN}, $Ranges{$ClusterID}{$Dimension}{$LINE_MAX})\n";

			print "___ Other Edges Values:\n______ ";
			foreach $OtherEdge (keys %{$Ranges{$ClusterID}{$Dimension}{$OTHER_EDGES}})
			{
				$val_at_min = $Ranges{$ClusterID}{$Dimension}{$OTHER_EDGES}{$OtherEdge}{$RANGE_MIN};
				$val_at_max = $Ranges{$ClusterID}{$Dimension}{$OTHER_EDGES}{$OtherEdge}{$RANGE_MAX};
				print "$OtherEdge($val_at_min,$val_at_max) ";
			}
			print "\n";
		}
	}
}

sub Get_NumClusters
{
    my ($hash_ptr) = @_;
	my $NumClusters = my @ClusterIDs = keys %{$hash_ptr};
	return $NumClusters;
}


sub Matches
{
	my ($hash_ptr1, $hash_ptr2) = @_;
	my %Dims1 = %{$hash_ptr1}, %Dims2 = %{$hash_ptr2};

	foreach $dim (keys %Dims1)
	{
		my $min1 = $Dims1{$dim}{$RANGE_MIN};
		my $max1 = $Dims1{$dim}{$RANGE_MAX};
		my $min2 = $Dims2{$dim}{$RANGE_MIN};
		my $max2 = $Dims2{$dim}{$RANGE_MAX};

		print "Checking match in $dim (fits $min2 in (".($min1 - $variation).", ".($min1 + $variation).") and $max2 in (".($max1 - $variation).", ".($max1 + $variation).")) ?? "; 

		$variation = $EdgesVariations{$dim}{$RANGE_VAR};
		
		if (($min2 < ($min1 - $variation)) || 
            ($min2 > ($min1 + $variation)) || 
            ($max2 < ($max1 - $variation)) || 
            ($max2 > ($max1 + $variation)))
		{
			print "NO\nDiscard Reason: Extremes didn't match ($dim)\n";
			return $FALSE;
		}
		else { print "YES\n"; }

#		foreach $OtherEdge (keys %{$Dims1{$dim}{$OTHER_EDGES}})
#		{
#			$variation = $EdgesVariations{$OtherEdge}{$RANGE_VAR};
#
#			my $val_at_min1 = $Dims1{$dim}{$OTHER_EDGES}{$OtherEdge}{$RANGE_MIN};
#			my $val_at_max1 = $Dims1{$dim}{$OTHER_EDGES}{$OtherEdge}{$RANGE_MAX};
#			my $val_at_min2 = $Dims2{$dim}{$OTHER_EDGES}{$OtherEdge}{$RANGE_MIN};
#			my $val_at_max2 = $Dims2{$dim}{$OTHER_EDGES}{$OtherEdge}{$RANGE_MAX};
#
#			print "Check OtherDim $OtherEdge (fits $val_at_min2 in (".($val_at_min1 - $variation).", ".($val_at_min1 + $variation).") and $val_at_max2 in (".($val_at_max1 - $variation).", ".($val_at_max1 + $variation).")) ?? ";
#
#			if (($val_at_min2 < ($val_at_min1 - $variation)) ||
#			    ($val_at_min2 > ($val_at_min1 + $variation)) ||
#			    ($val_at_max2 < ($val_at_max1 - $variation)) ||
#			    ($val_at_max2 > ($val_at_max1 + $variation)))
#			{
#				print "NO\nDiscard Reason: Dimension $OtherEdge didn't match\n";
#				return $FALSE;	
#			}
#			else { print "YES\n"; }
#		}
	}
	return $TRUE;
}

sub LoadMetric
{
    my ($RawFile, $MetricName) = @_;
    my @MetricDataArray = ();
	my %MetricDataHash = ();

    open RAW, "$RawFile" or die "Error opening file '$RawFile'\n$!\n";

    my @grep_data = grep (/^$MetricName/, <RAW>);
    chomp $grep_data[0];
    @MetricDataArray = split (/,/, $grep_data[0]);

    close RAW or die "Error closing file '$RawFile'\n$!\n";

	for (my $i=1; $i<=$#MetricDataArray; $i++)
	{
		$MetricDataHash{$i} = $MetricDataArray[$i];
	}
    return %MetricDataHash;
}

sub CompareRanges 
{
	my ($hash1_ptr, $MetricsFile1, $hash2_ptr, $MetricsFile2) = @_;
	my %Ranges1 = %{$hash1_ptr}, %Ranges2 = %{$hash2_ptr};

	%AlreadyMatched = ();
	@MatchPending1 = sort keys %Ranges1;
	@MatchPending2 = sort keys %Ranges2;

	foreach $CID1 (@MatchPending1)
	{
		foreach $CID2 (@MatchPending2)
		{
			print "Matches $CID1 with $CID2 ?\n";
			if (Matches($Ranges1{$CID1}, $Ranges2{$CID2}))
			{
				print ("Answer: YES\n");
				$AlreadyMatched{$CID1} = $CID2;
				@MatchPending2 = grep { $_ ne $CID2 } @MatchPending2; 
				last;
			}
			print ("Answer: NO\n");
		}
	}

	my %TotDur1 = LoadMetric($MetricsFile1, "% Total duration");
	my $SumDur1 = 0;
	foreach $CID1 (keys %AlreadyMatched)
	{
		$SumDur1 += $TotDur1{$CID1};
	}
	$SumDur1 *= 100;
	my %TotDur2 = LoadMetric($MetricsFile2, "% Total duration");
	my $SumDur2 = 0;
	foreach $CID2 (values %AlreadyMatched)
	{
		$SumDur2 += $TotDur2{$CID2};
	}
	$SumDur2 *= 100;

    foreach $CID1 (keys %AlreadyMatched)
    {
        print "$CID1 matched with $AlreadyMatched{$CID1}\n";
    }
	print "% Duration matched in 1st clustering = $SumDur1 %\n";
	print "% Duration matched in 2nd clustering = $SumDur2 %\n";

#	if (($SumDur1 >= $MIN_DURATION_PCT) && ($SumDur2 >= $MIN_DURATION_PCT)) { return $TRUE; }
#	else { return $FALSE };

	return (($SumDur1 + $SumDur2) / 2);
}

################
###   MAIN   ###
################

my $CSVFile1     = $ARGV[0].".clustered.csv";
my $CSVFile2     = $ARGV[1].".clustered.csv";
my $MetricsFile1 = $ARGV[0].".clusters_info.csv";
my $MetricsFile2 = $ARGV[1].".clusters_info.csv";

my %ClusterRanges1 = LoadRanges ($CSVFile1);
my %ClusterRanges2 = LoadRanges ($CSVFile2);

PrintRanges (\%ClusterRanges1);
print "=========================\n";
PrintRanges (\%ClusterRanges2);

#$equal = CompareRanges (\%ClusterRanges1, $MetricsFile1, \%ClusterRanges2, $MetricsFile2);
#print "Equal? $equal\n";
#exit $equal;

$PctEqual = CompareRanges (\%ClusterRanges1, $MetricsFile1, \%ClusterRanges2, $MetricsFile2);
print "Equal? $PctEqual\n";

#######################
### Generate graphs ###
#######################

my $STABILITY_DAT  = "STABILITY.dat";
my $STABILITY_PLOT = "STABILITY.gnuplot";

if (not -r $STABILITY_DAT)
{
    open  DAT, ">$STABILITY_DAT" or die "Error opening file '$STABILITY_DAT'\n$!\n";
    print DAT "Step,\%\n";
    close DAT or die "Error closing file '$STABILITY_DAT'\n$!\n";
}
$lines = `cat $STABILITY_DAT | wc -l`;
chomp $lines;
`echo "$lines,$PctEqual" >> $STABILITY_DAT`;

if (not -r $STABILITY_PLOT)
{
    open  PLOT, ">$STABILITY_PLOT" or die "Error opening file '$STABILITY_PLOT'\n$!\n";

    print PLOT  "set datafile separator \",\"\n".
                "set title \"Stability\"\n".
                "set style data boxes\n".
                "set style fill solid noborder\n".
                "set boxwidth 0.5\n".
                "set xlabel 'Steps'\n".
                "set xtics 1\n".
                "set ylabel '%'\n".
                "set yrange [0:100]\n".
                "plot \"$STABILITY_DAT\" using 1:2 noti\n".
                "pause -1 \"Hit return to continue...\"\n";

    close PLOT or die "Error closing file '$STABILITY_PLOT'\n$!\n";
}
`touch $STABILITY_PLOT`;

exit $PctEqual;
