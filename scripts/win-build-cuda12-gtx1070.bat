@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "BUILD_DIR=%ROOT_DIR%\build-cuda12-sm61"

pushd "%ROOT_DIR%"
if errorlevel 1 goto ERROR

call :setup_msvc_env
if errorlevel 1 goto ERROR

call :find_cuda12_nvcc
if errorlevel 1 goto ERROR

echo [INFO] Using nvcc: %CUDA_NVCC%
echo [INFO] Build dir : %BUILD_DIR%

:::  -DCMAKE_CUDA_ARCHITECTURES=61 for GTX 1070
::: adjust for other GPUs, e.g. 75 for 2080ti, 80 for RTX 4090, 86 for RTX 4080
cmake -S . -B "%BUILD_DIR%" ^
  -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON ^
  -DGGML_NATIVE=OFF ^
  -DCMAKE_CUDA_ARCHITECTURES=61 ^
  -DCMAKE_CUDA_COMPILER="%CUDA_NVCC%"
if errorlevel 1 goto ERROR

cmake --build "%BUILD_DIR%" -j %NUMBER_OF_PROCESSORS%
if errorlevel 1 goto ERROR

echo [OK] Build finished: %BUILD_DIR%
popd
exit /B 0

:setup_msvc_env
if defined VSCMD_VER (
  echo [INFO] MSVC dev environment already active.
  exit /B 0
)

set "FIXED_VCVARS=D:\s\VS2019\VC\Auxiliary\Build\vcvars64.bat"
if exist "%FIXED_VCVARS%" (
  echo [INFO] Using fixed MSVC env script: %FIXED_VCVARS%
  call "%FIXED_VCVARS%"
  if errorlevel 1 (
    echo [ERROR] Failed to initialize MSVC environment from fixed path.
    exit /B 1
  )
  exit /B 0
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo [ERROR] vswhere not found: %VSWHERE%
  echo [ERROR] Open this script from "x64 Native Tools Command Prompt for VS".
  exit /B 1
)

for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_PATH=%%I"

if not defined VS_PATH (
  echo [ERROR] Visual Studio with C++ build tools not found.
  exit /B 1
)

call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
  echo [ERROR] Failed to initialize MSVC environment.
  exit /B 1
)

exit /B 0

:find_cuda12_nvcc
set "CUDA_NVCC="

for %%V in (12_9 12_8 12_7 12_6 12_5 12_4 12_3 12_2 12_1 12_0) do (
  call set "CAND=%%CUDA_PATH_V%%V%%\bin\nvcc.exe"
  if defined CAND if exist "!CAND!" (
    set "CUDA_NVCC=!CAND!"
    goto CUDA_FOUND
  )
)

if defined CUDA_PATH (
  if exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo [WARN] CUDA_PATH is not explicitly CUDA 12; fallback to CUDA_PATH.
    set "CUDA_NVCC=%CUDA_PATH%\bin\nvcc.exe"
    goto CUDA_FOUND
  )
)

if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe" (
  set "CUDA_NVCC=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe"
  goto CUDA_FOUND
)

echo [ERROR] CUDA 12 nvcc not found.
echo [ERROR] Install CUDA 12 and ensure CUDA_PATH_V12_X is set.
exit /B 1

:CUDA_FOUND
exit /B 0

:ERROR
set "ERR=%ERRORLEVEL%"
echo [ERROR] Build failed with code %ERR%.
popd
exit /B %ERR%
