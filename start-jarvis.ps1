$root = "$env:USERPROFILE\jarvis"
Start-Process powershell -ArgumentList '-NoExit','-Command',"cd $root; `$env:PYTHONUTF8='1'; .\.venv\Scripts\Activate.ps1; python server.py --host 127.0.0.1"
Start-Process powershell -ArgumentList '-NoExit','-Command',"cd $root\frontend; npm run dev"
Start-Sleep 10
Start-Process chrome http://localhost:5173
