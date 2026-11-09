$root = "$env:USERPROFILE\jarvis"
# Il log del server finisce anche in data\server.log (sovrascritto a ogni avvio).
Start-Process powershell -ArgumentList '-NoExit','-Command',"cd $root; `$env:PYTHONUTF8='1'; `$env:PYTHONUNBUFFERED='1'; .\.venv\Scripts\Activate.ps1; cmd /c 'python server.py --host 127.0.0.1 2>&1' | Tee-Object -FilePath data\server.log"
Start-Process powershell -ArgumentList '-NoExit','-Command',"cd $root\frontend; npm run dev"
Start-Sleep 10
Start-Process chrome http://localhost:5173
