#!/usr/bin/perl

my $FALSE = 0;
my $TRUE = 1;
my $MIN_DURATION_THRESHOLD = 5; # Only plot clusters with this minimum % of ... ? */

my %RawMetricsData = ( );

sub LoadMetric
{
	my ($RawFile, $MetricName) = @_;
	my @MetricData;
	my $NumClusters;
	
	open RAW, "$RawFile" or die "Error opening file '$RawFile'\n$!\n";

	my @grep_data = grep (/^$MetricName/, <RAW>);
	chomp $grep_data[0];
	@MetricData = split (/,/, $grep_data[0]);
	$NumClusters = $#MetricData;
	
	$RawMetricsData{$MetricName} = [ @MetricData[1..$NumClusters] ];

	close RAW or die "Error closing file '$RawFile'\n$!\n";

	return $NumClusters;
}

# Returns a pointer to the array of values of the metric specified
sub GetMetric {
	my ($metric_name) = @_;
	return $RawMetricsData{$metric_name};
}

sub DivideMetrics {
	my ($data1_ptr, $data2_ptr) = @_;
	my @data1 = @{$data1_ptr}, @data2 = @{$data2_ptr};
	my @result;
	my $nc;

	$nc = @data1;
	for ($i=0; $i<$nc; $i++)
	{
		push(@result, $data1[$i] / $data2[$i]);
	}
	return @result;
}

sub AddMetrics {
    my ($data1_ptr, $data2_ptr) = @_;
    my @data1 = @{$data1_ptr}, @data2 = @{$data2_ptr};
    my @result;
    my $nc;

    $nc = @data1;
    for ($i=0; $i<$nc; $i++)
    {
        push(@result, $data1[$i] + $data2[$i]);
    }
    return @result;
}

sub SubstractMetrics {
	my ($data1_ptr, $data2_ptr) = @_;
	my @data1 = @{$data1_ptr}, @data2 = @{$data2_ptr};
	my @result;
    my $nc;

	$nc = @data1;
	for ($i=0; $i<$nc; $i++)
    {
		push(@result, $data1[$i] - $data2[$i]);
	}
	return @result;
}	

sub Pct {
	my ($data_ptr) = @_;
	my @data = @{$data_ptr};
	my @result;
	my $nc;
	
	$nc = @data;
    for ($i=0; $i<$nc; $i++)
    {
		my $value = sprintf("%.2f%", $data[$i] * 100);
        push(@result, $value);
    }
    return @result;
}
 
sub PrintMetric
{
	my ($fd, $metric_name, $data_ptr, $normalize, $delimiter) = @_;
	my @data = @{$data_ptr};
	my @values_array;

	$nc = @data;
	for ($i=0; $i<$nc; $i++)
	{
		my $value = 0;
		if ($normalize == $TRUE) 
		{
			$value = sprintf("%.2f%", $data[$i] * 100);
		}
		else 
		{
			$value = $data[$i];
		}
		push(@values_array, $value);
	}
	print $fd $metric_name.$delimiter;
	print $fd join($delimiter, @values_array)."\n";
}

sub WriteGNUplotScript
{
	my ($script_name, $data_file, $plot_title, $plot_mask_ptr) = @_;
	my @plot_mask = @{$plot_mask_ptr};

	open SCRIPT, ">$script_name" or die "Error creating file '$script_name'\n$!\n";

	print SCRIPT "set datafile separator \",\"\n";
	print SCRIPT "set title \"$plot_title\"\n"; 
	print SCRIPT "set style data histograms\n";
	print SCRIPT "set style histogram columnstacked\n";
	print SCRIPT "set style fill solid noborder\n";
	print SCRIPT "set key noinvert box right outside center\n";
	print SCRIPT "set yrange [0:100]\n";;
	print SCRIPT "set ylabel \"% of total\"\n";
	print SCRIPT "set xlabel \"Clusters\"\n";
	print SCRIPT "set tics scale 0.0\n";
	print SCRIPT "set boxwidth 0.75\n";
	print SCRIPT "set ytics\n";
	print SCRIPT "unset xtics\n";
	print SCRIPT "set xtics norotate nomirror\n";
	print SCRIPT "plot '$data_file' ";

	my $n_cols = @plot_mask;
	my $count_cols = 0;
	for (my $i=0; $i<$n_cols; $i++)
	{
		if ($plot_mask[$i] == 1) 
		{ 
			# Count how many columns will be plotted
			$count_cols ++; 
		}
	}

	if ($count_cols == 0) 
	{ 	
		# Plot them all
		$count_cols = $n_cols;
		for (my $i=0; $i<$n_cols; $i++) 
		{
			$plot_mask[$i] = 1;
		}
	}

	for (my $i=0; $i<$n_cols; $i++)
	{
		if ($plot_mask[$i] == 1)
		{
			$count_cols --;
			if ($count_cols == 0)
			{
				print SCRIPT "using ".($i+2)." : key(1) ti col\n";
			}
			else
			{
				print SCRIPT "using ".($i+2)." ti col, '' ";
			}
		}
	}
	print SCRIPT "pause -1 \"Press ENTER to continue...\"\n";

	close SCRIPT;
}

################
###   MAIN   ###
################

my $RawMetricsFile = $ARGV[0];

### Load raw metrics from raw data file ###

LoadMetric($RawMetricsFile, "Cluster Name");
LoadMetric($RawMetricsFile, "% Total duration");
LoadMetric($RawMetricsFile, "PM_CYC");
LoadMetric($RawMetricsFile, "PM_GRP_CMPL");
LoadMetric($RawMetricsFile, "PM_GCT_EMPTY_IC_MISS");
LoadMetric($RawMetricsFile, "PM_GCT_EMPTY_BR_MPRED");
LoadMetric($RawMetricsFile, "PM_GCT_EMPTY_CYC");
LoadMetric($RawMetricsFile, "PM_CMPLU_STALL_ERAT_MISS");
LoadMetric($RawMetricsFile, "PM_CMPLU_STALL_REJECT");
LoadMetric($RawMetricsFile, "PM_CMPLU_STALL_DCACHE_MISS");
LoadMetric($RawMetricsFile, "PM_CMPLU_STALL_LSU");
LoadMetric($RawMetricsFile, "PM_CMPLU_STALL_DIV");
LoadMetric($RawMetricsFile, "PM_CMPLU_STALL_FXU");
LoadMetric($RawMetricsFile, "PM_CMPLU_STALL_FDIV");
LoadMetric($RawMetricsFile, "PM_CMPLU_STALL_FPU");

### Calculate derived metrics ###

$PM_CYC_DATA_PTR = GetMetric("PM_CYC");
$Cluster_Names = GetMetric("Cluster Name");
$NumClusters = @{$Cluster_Names};

@Total_Group_Complete_Cycles = DivideMetrics(GetMetric("PM_GRP_CMPL"), $PM_CYC_DATA_PTR);

@Icache_Miss_Penalty = DivideMetrics(GetMetric("PM_GCT_EMPTY_IC_MISS"), $PM_CYC_DATA_PTR);
@Branch_Mispredict_Penalty = DivideMetrics(GetMetric("PM_GCT_EMPTY_BR_MPRED"), $PM_CYC_DATA_PTR);
@Total_GCT_Empty_Cycles = DivideMetrics(GetMetric("PM_GCT_EMPTY_CYC"), $PM_CYC_DATA_PTR);
@tmp = SubstractMetrics(\@Total_GCT_Empty_Cycles, \@Icache_Miss_Penalty);
@Other_GCT_Stalls = SubstractMetrics(\@tmp, \@Branch_Mispredict_Penalty);

@Stall_Xlate = DivideMetrics(GetMetric("PM_CMPLU_STALL_ERAT_MISS"), $PM_CYC_DATA_PTR);
@Total_Reject_Stall_Cycles = DivideMetrics(GetMetric("PM_CMPLU_STALL_REJECT"), $PM_CYC_DATA_PTR);
@Other_Reject = SubstractMetrics(\@Total_Reject_Stall_Cycles, \@Stall_Xlate);

@Stall_Dcache_Miss = DivideMetrics(GetMetric("PM_CMPLU_STALL_DCACHE_MISS"), $PM_CYC_DATA_PTR);
@Total_LSU_Stall_Cycles =DivideMetrics(GetMetric("PM_CMPLU_STALL_LSU"), $PM_CYC_DATA_PTR);
@tmp = SubstractMetrics(\@Total_LSU_Stall_Cycles, \@Total_Reject_Stall_Cycles);
@Stall_LSU_Basic_Latency = SubstractMetrics(\@tmp, \@Stall_Dcache_Miss);

@Stall_DIV = DivideMetrics(GetMetric("PM_CMPLU_STALL_DIV"), $PM_CYC_DATA_PTR);
@Total_FXU_Stall_Cycles = DivideMetrics(GetMetric("PM_CMPLU_STALL_FXU"), $PM_CYC_DATA_PTR);
@Stall_FXU_Basic_Latency = SubstractMetrics(\@Total_FXU_Stall_Cycles, \@Stall_DIV);

@Stall_FDIV = DivideMetrics(GetMetric("PM_CMPLU_STALL_FDIV"), $PM_CYC_DATA_PTR);
@Total_FPU_Stall_Cycles = DivideMetrics(GetMetric("PM_CMPLU_STALL_FPU"), $PM_CYC_DATA_PTR);
@Stall_FPU_Basic_Latency = SubstractMetrics(\@Total_FPU_Stall_Cycles, \@Stall_FDIV);

@Total_CSC = AddMetrics(\@Total_Group_Complete_Cycles, \@Total_GCT_Empty_Cycles);
for (my $i=0; $i<$NumClusters; $i++)
{
	$Total_CSC[$i] = 1 - $Total_CSC[$i];
}
@tmp = SubstractMetrics(\@Total_CSC, \@Total_LSU_Stall_Cycles);
@tmp = SubstractMetrics(\@tmp, \@Total_FXU_Stall_Cycles);
@Other_Stalls = SubstractMetrics(\@tmp, \@Total_FPU_Stall_Cycles);

### Select which clusters will be plotted depending on their total duration
my $Total_Duration_ptr = GetMetric("% Total duration");
my @Total_Duration = @{$Total_Duration_ptr};
my @Plotted_Clusters = ();
for (my $i=0; $i<$NumClusters; $i++)
{
	if ($Total_Duration[$i] * 100 > $MIN_DURATION_THRESHOLD)
	{
		push(@Plotted_Clusters, 1);
	}
	else
	{
		push(@Plotted_Clusters, 0);
	}
	
}

### Generate CPI stack general chart data ###

my $CPIGeneralFile = $RawMetricsFile;
$CPIGeneralFile =~ s/clusters_info.csv/cpistack_general.csv/g;
open CPI_GENERAL, ">$CPIGeneralFile" or die "Error opening file '$CPIGeneralFile'\n$!\n";
PrintMetric(CPI_GENERAL, "Cluster Name", $Cluster_Names, $FALSE, ",");
PrintMetric(CPI_GENERAL, "Total Group Complete Cycles", \@Total_Group_Complete_Cycles, $TRUE, ",");
PrintMetric(CPI_GENERAL, "Total GCT Empty Cycles", \@Total_GCT_Empty_Cycles, $TRUE, ",");
PrintMetric(CPI_GENERAL, "Total LSU Stall Cycles", \@Total_LSU_Stall_Cycles, $TRUE, ",");
PrintMetric(CPI_GENERAL, "Total FXU Stall Cycles", \@Total_FXU_Stall_Cycles, $TRUE, ",");
PrintMetric(CPI_GENERAL, "Total FPU Stall Cycles", \@Total_FPU_Stall_Cycles, $TRUE, ",");
PrintMetric(CPI_GENERAL, "Other Stalls", \@Other_Stalls, $TRUE, ",");
close CPI_GENERAL;

### Generate GNUplot general script ###

my $GNUplotGeneral = $CPIGeneralFile;
$GNUplotGeneral =~ s/cpistack_general.csv/cpistack_general.gnuplot/g;
WriteGNUplotScript($GNUplotGeneral, $CPIGeneralFile, "CPI Stack - General View", \@Plotted_Clusters);

### Generate CPI stack detailed chart data ###

my $CPIDetailFile = $RawMetricsFile;
$CPIDetailFile =~ s/clusters_info.csv/cpistack_detail.csv/g;
open CPI_DETAIL, ">$CPIDetailFile" or die "Error opening file '$CPIDetailFile'\n$!\n";
PrintMetric(CPI_DETAIL, "Cluster Name", $Cluster_Names, $FALSE, ",");
PrintMetric(CPI_DETAIL, "Total Group Complete Cycles", \@Total_Group_Complete_Cycles, $TRUE, ",");
PrintMetric(CPI_DETAIL, "I-cache Miss Penalty", \@Icache_Miss_Penalty, $TRUE, ",");
PrintMetric(CPI_DETAIL, "Branch Misspredict Penalty", \@Branch_Mispredict_Penalty, $TRUE, ",");
PrintMetric(CPI_DETAIL, "Other GCT Stalls", \@Other_GCT_Stalls, $TRUE, ",");
PrintMetric(CPI_DETAIL, "Stall by Xlate", \@Stall_Xlate, $TRUE, ",");
PrintMetric(CPI_DETAIL, "Other Reject", \@Other_Reject, $TRUE, ",");
PrintMetric(CPI_DETAIL, "Stall by D-cache Miss", \@Stall_Dcache_Miss, $TRUE, ",");
PrintMetric(CPI_DETAIL, "Stall by LSU Basic Latency", \@Stall_LSU_Basic_Latency, $TRUE, ",");
PrintMetric(CPI_DETAIL, "Stall by Div/MTSPR/MFSPR", \@Stall_DIV, $TRUE, ",");
PrintMetric(CPI_DETAIL, "Stall by FXU Basic Latency", \@Stall_FXU_Basic_Latency, $TRUE, ",");
PrintMetric(CPI_DETAIL, "Stall by FDIV/FSQRT", \@Stall_FDIV, $TRUE, ",");
PrintMetric(CPI_DETAIL, "Stall by FPU Basic Latency", \@Stall_FPU_Basic_Latency, $TRUE, ",");
PrintMetric(CPI_DETAIL, "Other Stalls", \@Other_Stalls, $TRUE, ",");
close CPI_DETAIL;

### Generate GNUplot detailed script ###

my $GNUplotDetail = $CPIDetailFile;
$GNUplotDetail =~ s/cpistack_detail.csv/cpistack_detail.gnuplot/g;
WriteGNUplotScript($GNUplotDetail, $CPIDetailFile, "CPI Stack - Detailed View", \@Plotted_Clusters);

### Print data ###

PrintMetric(STDOUT, "Cluster Name", $Cluster_Names, $FALSE, ",");
PrintMetric(STDOUT, "Total Group Complete Cycles", \@Total_Group_Complete_Cycles, $TRUE, ",");
PrintMetric(STDOUT, "I-cache Miss Penalty", \@Icache_Miss_Penalty, $TRUE, ",");
PrintMetric(STDOUT, "Branch Misspredict Penalty", \@Branch_Mispredict_Penalty, $TRUE, ",");
PrintMetric(STDOUT, "Other GCT Stalls", \@Other_GCT_Stalls, $TRUE, ",");
PrintMetric(STDOUT, "Total GCT Empty Cycles", \@Total_GCT_Empty_Cycles, $TRUE, ",");
PrintMetric(STDOUT, "Stall by Xlate", \@Stall_Xlate , $TRUE, ",");
PrintMetric(STDOUT, "Other Reject", \@Other_Reject, $TRUE, ",");
PrintMetric(STDOUT, "Total Reject Stall Cycles", \@Total_Reject_Stall_Cycles, $TRUE, ",");
PrintMetric(STDOUT, "Stall by D-cache Miss", \@Stall_Dcache_Miss, $TRUE, ",");
PrintMetric(STDOUT, "Stall by LSU Basic Latency", \@Stall_LSU_Basic_Latency, $TRUE, ",");
PrintMetric(STDOUT, "Total LSU Stall Cycles", \@Total_LSU_Stall_Cycles, $TRUE, ",");
PrintMetric(STDOUT, "Stall by Div/MTSPR/MFSPR", \@Stall_DIV, $TRUE, ",");
PrintMetric(STDOUT, "Stall by FXU Basic Latency", \@Stall_FXU_Basic_Latency, $TRUE, ",");
PrintMetric(STDOUT, "Total FXU Stall Cycles", \@Total_FXU_Stall_Cycles, $TRUE, ",");
PrintMetric(STDOUT, "Stall by FDIV/FSQRT", \@Stall_FDIV, $TRUE, ",");
PrintMetric(STDOUT, "Stall by FPU Basic Latency", \@Stall_FPU_Basic_Latency, $TRUE, ",");
PrintMetric(STDOUT, "Total FPU Stall Cycles", \@Total_FPU_Stall_Cycles, $TRUE, ",");
PrintMetric(STDOUT, "Other Stalls", \@Other_Stalls, $TRUE, ",");
PrintMetric(STDOUT, "Total CSC", \@Total_CSC, $TRUE, ",");

