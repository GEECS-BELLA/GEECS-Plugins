SET "ANACONDA_Xopt_GEECS=C:\Users\loasis.LOASIS\AppData\Local\anaconda3\envs\Xopt-GEECS"
SET "GEECS_Plugins_Directory=C:\GEECS\Developers Version\source\GEECS-Plugins"
:: SET "GEECS_Plugins_Directory=C:\Users\loasis.LOASIS\Documents\GitHub\GEECS-Plugins"

git clone https://github.com/slaclab/Badger-Plugins.git "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins"

mklink %ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\environments\geecs "%GEECS_Plugins_Directory%\Xopt-GEECS\badger-plugins\environments\geecs"
mklink "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\environments\camera_exposure_time_test" "%GEECS_Plugins_Directory%\Xopt-GEECS\badger-plugins\environments\camera_exposure_time_test"
mklink "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\environments\HTU_hex_alignment_sim" "%GEECS_Plugins_Directory%\Xopt-GEECS\badger-plugins\environments\HTU_hex_alignment_sim"
mklink "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\interfaces\geecs" "%GEECS_Plugins_Directory%\Xopt-GEECS\badger-plugins\interfaces\geecs"
pause
