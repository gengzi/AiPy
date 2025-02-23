
Unicode true


!include "MUI2.nsh"


Name "Llama CLI Installer"
OutFile "LlamaCLIInstaller.exe"
InstallDir "$PROGRAMFILES\LlamaCLI"

!define MUI_PAGE_WELCOME
!define MUI_PAGE_DIRECTORY
!define MUI_PAGE_INSTFILES


!define MUI_UNPAGE_WELCOME
!define MUI_UNPAGE_CONFIRM
!define MUI_UNPAGE_INSTFILES

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English" ; 可根据需要修改语言


Section "Install"
    SetOutPath "$INSTDIR"
    File "E:\ruanjian\llama-b4751-bin-win-avx-x64\llama-server.exe"
    File "E:\ruanjian\llama-b4751-bin-win-avx-x64\ggml.dll"
    File "E:\ruanjian\llama-b4751-bin-win-avx-x64\ggml-base.dll"
    File "E:\ruanjian\llama-b4751-bin-win-avx-x64\ggml-cpu.dll"
    File "E:\ruanjian\llama-b4751-bin-win-avx-x64\ggml-rpc.dll"
    File "E:\ruanjian\llama-b4751-bin-win-avx-x64\llama.dll"
    File "E:\models\ollama\DeepSeek-R1-Distill-Qwen-1.5B-NexaQuant.gguf"
    ExecWait '"$INSTDIR\llama-server.exe" -m "$INSTDIR\DeepSeek-R1-Distill-Qwen-1.5B-NexaQuant.gguf"'
SectionEnd


Section "Uninstall"
    Delete "$INSTDIR\llama-cli.exe"
    Delete "$INSTDIR\DeepSeek-R1-Distill-Qwen-1.5B-NexaQuant.gguf"
    RMDir "$INSTDIR"
SectionEnd