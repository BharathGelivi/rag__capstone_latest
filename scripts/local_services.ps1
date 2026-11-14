param(
    [ValidateSet('start', 'stop', 'status')]
    [string]$Action = 'start'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$logRoot = Join-Path $projectRoot '.cache\local-services'
$statePath = Join-Path $logRoot 'processes.json'
$services = @(
    @{ Name = 'api'; Port = 8010; Url = 'http://127.0.0.1:8010/ui/config' },
    @{ Name = 'dashboard'; Port = 5173; Url = 'http://127.0.0.1:5173' },
    @{ Name = 'viewer'; Port = 8085; Url = 'http://127.0.0.1:8085/explorer.html' }
)
$records = @()
if (Test-Path -LiteralPath $statePath) {
    $loaded = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $records = @(foreach ($record in $loaded) { $record })
}

if ($Action -eq 'stop') {
    foreach ($record in $records) {
        $process = Get-Process -Id $record.Id -ErrorAction SilentlyContinue
        # Match start time as well as PID so a reused PID is never stopped.
        if ($process -and $process.StartTime.ToUniversalTime().ToString('o') -eq $record.StartTime) {
            # The Windows venv launcher can have a Python child; stop its tree.
            & "$env:SystemRoot\System32\taskkill.exe" /PID $record.Id /T /F | Out-Null
            Write-Output "Stopped $($record.Name) (PID $($record.Id))."
        }
    }
    if (Test-Path -LiteralPath $statePath) { Remove-Item -LiteralPath $statePath }
    return
}

if ($Action -eq 'start') {
    $pythonExe = Join-Path $projectRoot 'venv\Scripts\python.exe'
    $viteScript = Join-Path $projectRoot 'frontend\node_modules\vite\bin\vite.js'
    $viewerScript = Join-Path $env:USERPROFILE '.codex\skills\codecharta\scripts\serve.mjs'
    $nodeExe = (Get-Command node.exe -ErrorAction Stop).Source
    foreach ($path in @($pythonExe, $viteScript)) {
        if (!(Test-Path -LiteralPath $path)) { throw "Missing prerequisite: $path. See README startup instructions." }
    }
    if (!(Test-Path -LiteralPath $viewerScript) -or !(Test-Path -LiteralPath (Join-Path $projectRoot '.codecharta\explorer.html'))) {
        $services = @($services | Where-Object Name -ne 'viewer')
        Write-Output 'Skipping optional local source viewer; dashboard and API will start normally.'
    }
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    $previousDebug = $env:DEBUG
    $env:DEBUG = 'false'
    try {
        foreach ($service in $services) {
            $record = $records | Where-Object Name -eq $service.Name | Select-Object -First 1
            if ($record) {
                $existing = Get-Process -Id $record.Id -ErrorAction SilentlyContinue
                if ($existing -and $existing.StartTime.ToUniversalTime().ToString('o') -eq $record.StartTime) {
                    Write-Output "$($service.Name): recorded process is running (PID $($record.Id)); allowing startup to finish."
                    continue
                }
            }
            $listener = Get-NetTCPConnection -LocalPort $service.Port -State Listen -ErrorAction SilentlyContinue
            if ($listener) {
                Write-Output "$($service.Name): port $($service.Port) already listening; leaving existing process running."
                continue
            }
            $exe = $nodeExe
            $workingDirectory = $projectRoot
            switch ($service.Name) {
                'api' { $exe = $pythonExe; $arguments = @('run_api_ui.py') }
                'dashboard' {
                    $workingDirectory = Join-Path $projectRoot 'frontend'
                    $arguments = @('"' + $viteScript + '"', '--host', '127.0.0.1', '--port', '5173', '--strictPort')
                }
                'viewer' { $arguments = @('"' + $viewerScript + '"', '"' + $projectRoot + '"', '8085') }
            }
            $process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $workingDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot "$($service.Name).out.log") -RedirectStandardError (Join-Path $logRoot "$($service.Name).err.log")
            $records = @($records | Where-Object Name -ne $service.Name)
            $records += [pscustomobject]@{ Name = $service.Name; Id = $process.Id; StartTime = $process.StartTime.ToUniversalTime().ToString('o') }
            $records | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statePath -Encoding UTF8
            Write-Output "Started $($service.Name) (PID $($process.Id))."
        }
    } finally { $env:DEBUG = $previousDebug }
}

foreach ($service in $services) {
    try {
        $response = Invoke-WebRequest -Uri $service.Url -UseBasicParsing -TimeoutSec 5
        Write-Output "$($service.Name): HTTP $($response.StatusCode) $($service.Url)"
    } catch {
        Write-Output "$($service.Name): not ready at $($service.Url); check .cache/local-services/$($service.Name).err.log or run status again."
    }
}
