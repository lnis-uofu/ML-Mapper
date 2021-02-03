#!/usr/bin/perl -w
##################################################################
### Generates train model for a given circuit in a given tech
##################################################################

use strict; 
use warnings; 
use File::Copy;
use File::Basename;
use POSIX;

# needs to pass blif or aig + lib file  
my $number_arguments = @ARGV;
if ( $number_arguments < 1 ) {
	print(
		"usage: train.pl <circuit_file> <libery_file>\n"
	);
	exit(-1);
}
my $ckt = shift(@ARGV);
my $lib_file = shift(@ARGV);
my $rounds = 200; 
my $epochs = 50; 
my $train = 0.7;
my $validation = 0.3;
my $model = "cnn"; 
my $infModel = "rc16b";
my $classes = 10; 
my $inference = 0; 

# change your path here
my $csv_data = '/home/walterl/cleanLearning/data/csv/'; 
my $csv_work = '/home/walterl/cleanLearning/data/work/'; 
my $nn_work = '/home/walterl/cleanLearning/nn/work/'; 

# parses arguments
while ( scalar(@ARGV) != 0 ) { #While non-empty
  my $token = shift(@ARGV);
  if ( $token eq "-rounds" ) {
		$rounds = shift(@ARGV);
  } elsif ( $token eq "-epochs" ) {
		$epochs = shift(@ARGV);
  }
  elsif ( $token eq "-train" ) {
		$train = shift(@ARGV);
  }
  elsif ( $token eq "-validation" ) {
		$validation = shift(@ARGV);
  }
  elsif ( $token eq "-model" ) {
		$model = shift(@ARGV);
  }
  elsif ( $token eq "-classes" ) {
		$model = shift(@ARGV);
  }
  elsif ( $token eq "-inference" ) {
		$inference = shift(@ARGV);
  }
}

print "parameters: $ckt $lib_file $rounds $epochs $train $validation $model\n";

# set-up filenames 
my @name = split('\.', $ckt);
my $cktName = $name[0];
  
# generates node embedding and features for inference
#my @dummy = `./abc-train -c "read_lib -v $lib_file; r $ckt; st; prepare_map -f $cut_table -F $feat -n $embed; q"`;

my $filename = $cktName . '_train_hashed.csv'; 
my %visited;
open(FH, '>', $filename) or die $!; 

my $feat = $cktName . '_feat_sweep.csv'; 
my $embed = $cktName . '_node_embed.csv'; 
my $cut_table = $cktName . '_cut_table.csv'; 

for ( my $i = 1; $i <= $rounds ; $i++ )
{
  my @output = `./abc-train -c "read_lib -v $lib_file; r $ckt; st; map; topo; stime; q"`;
  #print "dbg: $output[$#output]";
  my @fields = split(', ', $output[$#output]);
  #print "dbg: @fields";
  #print "dbg: $fields[2], $fields[3]";
  my $toHash = $fields[2] * $fields[3];
  #print "to hash value is $toHash\n";
  if (exists $visited{$toHash}) {
    print "QoR already seen $toHash\n";
  }
  else {
      $visited{$toHash} = 1;
      print FH @output; 
    }
    print "$ckt iteration $i/$rounds\n";
  }

  my $target_train = $csv_data . $filename;
  my $target_embed = $csv_work . $embed; 
  my $target_feat = $csv_work . $feat; 
  my $train_file = $csv_work . $filename; 

  system("cp $filename $target_train"); 
  system("cp $embed $target_embed"); 
  system("cp $feat $target_feat"); 

  chdir($csv_work) or die "$!";
  my @genCSV = `./genCSV.sh $target_train $train_file`;

  chdir($nn_work) or die "$!";
  print "Train file is $train_file\n";

  my $dataPoints = `wc -l < $train_file`;
  my $trainPoints = floor($train * $dataPoints);
  my $valPoints = floor($validation * $dataPoints); 

  print "Training points are $trainPoints, validation points are $valPoints\n";

  if ( $model eq "cnn") { 
    # set configurations on the shell script
    `sed -i 's|##train_path##|$train_file|g' run-cnn-train.sh`;
    `sed -i 's/##classes##/$classes/g' run-cnn-train.sh`;
    `sed -i 's/##train_points##/$trainPoints/g' run-cnn-train.sh`;
    `sed -i 's/##validation_points##/$valPoints/g' run-cnn-train.sh`;
    `sed -i 's/##epochs##/$epochs/g' run-cnn-train.sh`;
    # calls the shell script
    system("sh", "run-cnn-train.sh");
    # reverts configuration
    `sed -i 's/$classes/##classes##/g' run-cnn-train.sh`;
    `sed -i 's/$trainPoints/##train_points##/g' run-cnn-train.sh`;
    `sed -i 's/$valPoints/##validation_points##/g' run-cnn-train.sh`;
    `sed -i 's/$epochs/##epochs##/g' run-cnn-train.sh`;
    `sed -i 's|$train_file|##train_path##|g' run-cnn-train.sh`;
  }
