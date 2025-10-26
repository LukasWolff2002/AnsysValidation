echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v252\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v252\fluent\ntbin\win64\tell.exe" Frodo 65418 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v252\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="Frodo" (%KILL_CMD% 8428) 
if /i "%LOCALHOST%"=="Frodo" (%KILL_CMD% 26432) 
if /i "%LOCALHOST%"=="Frodo" (%KILL_CMD% 20084) 
if /i "%LOCALHOST%"=="Frodo" (%KILL_CMD% 8084) 
if /i "%LOCALHOST%"=="Frodo" (%KILL_CMD% 24992) 
if /i "%LOCALHOST%"=="Frodo" (%KILL_CMD% 24160)
del "C:\Users\lkwol\AppData\Local\Temp\WB_lkwol_24832_2\wbnew_files\dp0\SYS\DM\cleanup-fluent-Frodo-24992.bat"
