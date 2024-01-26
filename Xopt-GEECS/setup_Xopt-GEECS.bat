SET "ANACONDA_Xopt_GEECS=C:\Users\loasis.LOASIS\AppData\Local\anaconda3\envs\Xopt-GEECS"

git clone https://github.com/slaclab/Badger-Plugins.git "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins"

mklink %ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\environments\geecs "C:\GEECS\Developers Version\source\GEECS-Plugins\Xopt-GEECS\badger-plugins\environments\geecs"
mklink "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\environments\camera_exposure_time_test" "C:\GEECS\Developers Version\source\GEECS-Plugins\Xopt-GEECS\badger-plugins\environments\camera_exposure_time_test"
mklink "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\environments\HTU_hex_alignment_sim" "C:\GEECS\Developers Version\source\GEECS-Plugins\Xopt-GEECS\badger-plugins\environments\HTU_hex_alignment_sim"
mklink "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\interfaces\geecs" "C:\GEECS\Developers Version\source\GEECS-Plugins\Xopt-GEECS\badger-plugins\interfaces\geecs"
pause


