param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)
& "$PSScriptRoot\venv\Scripts\python.exe" "$PSScriptRoot\web_server.py" @ScriptArgs
