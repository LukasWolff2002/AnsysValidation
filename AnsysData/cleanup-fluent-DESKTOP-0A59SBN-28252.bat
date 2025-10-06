echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v252\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v252\fluent\ntbin\win64\tell.exe" DESKTOP-0A59SBN 58555 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v252\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 25444) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 28488) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 25936) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 24984) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 28252) 
if /i "%LOCALHOST%"=="DESKTOP-0A59SBN" (%KILL_CMD% 29528)
del "C:\Users\MBX\Desktop\AnsysValidation\AnsysData\cleanup-fluent-DESKTOP-0A59SBN-28252.bat"
