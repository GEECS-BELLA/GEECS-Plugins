cd Z:\\
#cd '.\GEECS\Developers Version\source\GEECS-Plugins\LogMaker4GoogleDocs\logmaker_4_googledocs\'
cd 'Z:\software\control-all-loasis\HTU\Active Version\GEECS-Plugins\LogMaker4GoogleDocs\logmaker_4_googledocs'
pwd
poetry run python createGdoc.py kHzparameters.ini kHzplaceholders.ini kHzcurrentvalues.in

END=8640
for g in $(seq 1 $END)
do
sleep 10
poetry run python appendScan.py kHzparameters.ini kHzplaceholders.ini kHzcurrentvalues.ini
done