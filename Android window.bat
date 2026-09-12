@echo off
REM Script for connecting to Android via Wi-Fi (with USB handling and diagnostics)

REM Parameters
set "IP=10.10.10.196"
set "PORT=5555"
set "SCRCPY_OPTIONS=-e --video-bit-rate 2M --video-codec h264 --max-size 800 --max-fps=60 --disable-screensaver --keyboard=disabled"

echo === Android Wi-Fi Debugging (Wireless mode) ===

REM Step 1: Check device availability by IP
echo Step 1: Checking network reachability for %IP% ...
ping -n 1 %IP%
if errorlevel 1 (
    echo ERROR: Device did not respond to ping (%IP%)
    echo Please check:
    echo 1. Device and PC are on the same Wi-Fi network
    echo 2. Wireless debugging is enabled on your device
    echo 3. The IP address is correct (current: %IP%)
    REM Do not exit, allow further diagnostics
) else (
    echo ✓ Device is reachable on the network
)

REM Step 1.5: Check for USB connection and switch to tcpip
echo Step 1.5: Checking for USB connections...
set "USB_FOUND=0"
for /f "skip=1 tokens=1,2" %%a in ('adb\adb.exe devices') do (
    if not "%%a"=="%IP%:%PORT%" (
        if "%%b"=="device" (
            set "USB_FOUND=1"
            echo ⚠ USB connection detected: %%a
            echo Switching USB connection to Wi-Fi mode...
            adb\adb.exe -s %%a tcpip %PORT%
            timeout /t 3 >nul
        )
    )
)
if "%USB_FOUND%"=="0" (
    echo No USB devices detected.
)

REM Step 2: Connect via IP
echo Step 2: Connecting with ADB over Wi-Fi...
adb\adb.exe disconnect %IP%:%PORT% >nul
set "CONNECT_RESULT="
for /f "delims=" %%c in ('adb\adb.exe connect %IP%:%PORT%') do set "CONNECT_RESULT=%%c"
echo %CONNECT_RESULT% | findstr /i "connected" >nul
if not errorlevel 1 (
    echo ✓ %CONNECT_RESULT%
) else (
    echo %CONNECT_RESULT% | findstr /i "already connected" >nul
    if not errorlevel 1 (
        echo ⚠ %CONNECT_RESULT%
    ) else (
        echo ERROR: %CONNECT_RESULT%
        echo Attempting fix: restarting ADB server...
        adb\adb.exe kill-server
        adb\adb.exe start-server
        timeout /t 2 >nul
        set "CONNECT_RESULT="
        for /f "delims=" %%d in ('adb\adb.exe connect %IP%:%PORT%') do set "CONNECT_RESULT=%%d"
        echo %CONNECT_RESULT% | findstr /i "connected" >nul
        if errorlevel 1 (
            echo ERROR: Could not connect with ADB after restarting server.
        ) else (
            echo ✓ %CONNECT_RESULT%
        )
    )
)

REM Step 3: Check connection status
echo Step 3: Checking connection status...
set "STATE="
for /f "tokens=2" %%e in ('adb\adb.exe devices ^| findstr %IP%:%PORT%') do set "STATE=%%e"
if "%STATE%"=="device" (
    echo ✓ Device is ready
) else if "%STATE%"=="offline" (
    echo ERROR: Device is offline
    echo Try:
    echo 1. Restart the device
    echo 2. Disable and re-enable wireless debugging
) else if "%STATE%"=="unauthorized" (
    echo ERROR: Device not authorized
    echo Check debugging authorization on the device
) else (
    echo ERROR: Unable to connect (state: %STATE%)
    echo Attempting fix: resetting connection...
    adb\adb.exe disconnect %IP%:%PORT%
    timeout /t 2 >nul
    adb\adb.exe connect %IP%:%PORT% >nul
    set "STATE=None"
    for /f "tokens=2" %%f in ('adb\adb.exe devices ^| findstr %IP%:%PORT%') do set "STATE=%%f"
    if "%STATE%"=="device" (
        echo ✓ Device is ready
    ) else (
        echo ERROR: Could not recover connection (final state: %STATE%)
    )
)

REM Step 4: Launch scrcpy
echo Step 4: Launching scrcpy...
scrcpy\scrcpy.exe %SCRCPY_OPTIONS%
if errorlevel 1 (
    echo ERROR: Failed to launch scrcpy
    echo Check your scrcpy installation
) else (
    echo ✓ scrcpy launched successfully
)

REM Finish
adb\adb.exe disconnect %IP%:%PORT%
echo === Done ===
pause
