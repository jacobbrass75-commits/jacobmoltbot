# Create desktop shortcut for Moltbot Control Panel

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Moltbot.lnk")
$Shortcut.TargetPath = "$PSScriptRoot\..\venv\Scripts\pythonw.exe"
$Shortcut.Arguments = "`"$PSScriptRoot\..\gui.py`""
$Shortcut.WorkingDirectory = "$PSScriptRoot\.."
$Shortcut.Description = "Moltbot Control Panel"
$Shortcut.Save()

Write-Host "Desktop shortcut created: Moltbot.lnk"
