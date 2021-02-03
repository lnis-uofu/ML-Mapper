#!/bin/bash
inputFile=`readlink -f $1`
outputFile=`readlink -f $2`
echo "Reading in $inputFile"
mkdir -p .work
cd .work
rm -rf *
cp $inputFile .
localFile=`echo $inputFile | sed 's#.*/##g'`
csplit $localFile '/^ABC.*$/' '{*}' --quiet
rm xx00
echo "nodeid,fon,lvln,invn,invp1,lvlp1,fop1,invp2,lvlp2,fop2,tt,invc,leavesc,volumec,mincutlvl,maxcutlvl,cutlvl,cutminfo,cutmaxfo,cutfo,cutidx,relativelvl,l1id,l2id,l3id,l4id,l5id,numgates,cap,area,delay" > $outputFile
for splitFile in `find . -name "xx*"`
do
	grep -E '^[0-9]+,[0-9]+,.*' $splitFile | sed 's#,[^,]\+$##g' > splitFileFeature.tmp
	stime=`grep 'stime:' $splitFile | sed 's#stime\:##g'`
	splitFileFeatureSize=`wc -l splitFileFeature.tmp | awk '{print $1}'`
	echo "  Reading $splitFile with $splitFileFeatureSize datapoints"
	yes ",${stime}" | head -n ${splitFileFeatureSize} > splitFileLabel.tmp
	paste splitFileFeature.tmp splitFileLabel.tmp > splitFileData.tmp
	sed -i 's#[\t ]\+##g' splitFileData.tmp
	cat splitFileData.tmp >> $outputFile
done
cd ..
rm -rf .work
