echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v252\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v252\fluent\ntbin\win64\tell.exe" DESKTOP-0A59SBN 54968 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v252\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 31620) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 31052) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 29396) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 10156) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 34648) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 33512)
del "C:\Users\MBX\Desktop\AnsysValidation\AnsysData\cleanup-fluent-DESKTOP-0A59SBN-34648.bat"
