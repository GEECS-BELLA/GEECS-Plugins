setlocal

SET ANACONDA_Xopt_GEECS="C:\Users\loasis.LOASIS\AppData\Local\anaconda3\envs\Xopt-GEECS"
SET PYTHON_SCRIPT=generate_symlink_bat.py

CALL python "%python_script%" %ANACONDA_Xopt_GEECS%
CALL autosetup_Xopt-GEECS.bat

endlocal
