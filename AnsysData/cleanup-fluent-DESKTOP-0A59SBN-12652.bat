echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v252\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v252\fluent\ntbin\win64\tell.exe" DESKTOP-0A59SBN 62356 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v252\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 25840) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 38836) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 37660) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 32300) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 12652) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 25584)
del "C:\Users\MBX\Desktop\AnsysValidation\AnsysData\cleanup-fluent-DESKTOP-0A59SBN-12652.bat"
