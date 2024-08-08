@ECHO OFF
ECHO [Info] Checking admin rights
NET SESSION
CALL :CHECK_ERROR_ECHO [Error] You need admin rights to run this batch file
IF %ERRORLEVEL% NEQ 0 (
	TIMEOUT 5
	EXIT
)

ECHO [Info] Checking if chocolatey is installed...
choco --version
IF %ERRORLEVEL% NEQ 0 (
	CALL :INSTALL_CHOCOLATEY
	CALL :CHECK_ERROR_ECHO [Error] Could not install Chocolatey properly.
	TIMEOUT 5
	EXIT
)
ECHO [Info] Chocolatey already installed! =D

Echo [Info] Now we are doing all boring steps to get Tesseract up and running!

cd %USERPROFILE%\Downloads
choco install wget 7zip.install opencv -y
refreshenv
wget https://download.lfd.uci.edu/pythonlibs/l8ulg3xw/opencv_python-3.4.2-cp37-cp37m-win_amd64.whl
pip install "opencv_python-3.4.2-cp37-cp37m-win_amd64.whl"
mkdir "C:\Program Files\Tesseract-OCR\tessdata"
cd "C:\Program Files\Tesseract-OCR"
wget https://github.com/parrot-office/tesseract/releases/download/3.5.1/tesseract-Win64.zip
7z e tesseract-Win64.zip
del tesseract-Win64.zip
cd "C:\Program Files\Tesseract-OCR\tessdata"
wget https://github.com/tesseract-ocr/tessdata/raw/3.04.00/por.traineddata
setx PATH "%PATH%;C:\Program Files\Tesseract-OCR" /m
pip install pytesseract numpy pillow
EXIT /B %ERRORLEVEL%

:CHECK_ERROR_ECHO
IF %ERRORLEVEL% NEQ 0 (
	ECHO %*
)
EXIT /B %ERRORLEVEL%

:INSTALL_CHOCOLATEY
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
refreshenv
EXIT /B %ERRORLEVEL%