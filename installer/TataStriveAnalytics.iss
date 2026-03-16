#define MyAppName "TataStrive Analytics"
#define MyAppExeName "TataStriveAnalytics.exe"
#define MyAppVersion "1.0.0"

[Setup]
AppId={{F0DD6161-A6C7-4A40-8A62-8806902075E1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=TataStrive
DefaultDirName={autopf}\TataStrive Analytics
DefaultGroupName=TataStrive Analytics
DisableProgramGroupPage=yes
OutputDir=..\release
OutputBaseFilename=TataStriveAnalytics_Setup_{#MyAppVersion}
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
PrivilegesRequired=admin
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"

[Files]
Source: "..\dist\TataStriveAnalytics\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
